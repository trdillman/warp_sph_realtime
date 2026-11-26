"""Actual CUDA replay copy preservation; no fabricated solver or completion state."""
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import traceback

import bpy
import gpu
import numpy as np
import warp as wp


def run(output_path, consumer_name, expected_root, *, inject_after_submit=False):
    output=Path(output_path);assert output.is_absolute() and not output.exists()
    assert not bpy.app.background and bpy.context.window is not None
    assert bpy.app.build_hash.decode()=='fbe6228777e7' and gpu.platform.backend_type_get()=='VULKAN'
    assert wp.__version__=='1.17.0'
    assert '.verification-temp' in Path(bpy.utils.user_resource('CONFIG')).parts
    package=importlib.import_module(consumer_name)
    assert Path(package.__file__).resolve().parent==Path(expected_root).resolve()
    module=importlib.import_module(consumer_name+'.cache.solver_replay')
    core=importlib.import_module(consumer_name+'.sim.sim_core')
    descmod=importlib.import_module(consumer_name+'.scene.description')
    Solver=core.RealtimeWCSPHSolver
    report={'status':'running','pid':os.getpid(),'consumer_origin':package.__file__,
            'replay_sha256':hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
            'backend':'VULKAN','warp':wp.__version__,'checks':[],
            'scope':'actual independent solver clone; no native animation/UI acceptance implied'}
    original_frame=bpy.context.scene.frame_current
    scene_sizes=(len(bpy.data.objects),len(bpy.data.meshes),len(bpy.data.node_groups))
    replay=reference=cancelled=None
    def check(name,condition,**detail):
        assert condition,name
        report['checks'].append(dict(name=name,passed=True,**detail))
    def box(lo,hi,name):
        vertices=np.array([[(hi if i>>k&1 else lo)[k] for k in range(3)] for i in range(8)],dtype=np.float32)
        faces=np.array([[0,2,3],[0,3,1],[4,5,7],[4,7,6],[0,1,5],[0,5,4],[2,6,7],[2,7,3],[0,4,6],[0,6,2],[1,3,7],[1,7,5]],dtype=np.int32)
        return descmod.MeshData(vertices,faces,name)
    def digest(solver):
        wp.synchronize_device('cuda:0')
        arrays={name:hashlib.sha256(value.numpy().tobytes()).hexdigest()
                for name,value in vars(solver).items() if isinstance(value,wp.array)}
        return dict(arrays=arrays,sim_time=solver.sim_time,dt=solver.dt,substeps=solver.substeps,
                    next_id=solver._next_id,active_count=solver.active_count,
                    emitter_state=[dataclasses.asdict(e) for e in solver._emitters],
                    key_frames=dict(solver._key_frames),grid_identity=solver.grid.id,
                    cuda_graph_identity=id(solver.cuda_graph))
    try:
        nozzle=descmod.MeshData(np.array([[-.12,-.12,1.4],[.12,-.12,1.4],[.12,.12,1.4],[-.12,.12,1.4]],dtype=np.float32),np.array([[0,1,2],[0,2,3]],dtype=np.int32),'replay-nozzle')
        desc=descmod.SceneDescription((-.5,-.5,0.),(.5,.5,2.),(0.,0.,-9.81),
              (box((-.12,-.12,.65),(.12,.12,.9),'fluid'),),
              (box((-.3,-.3,.25),(.3,.3,.3),'plate'),),
              emitters=(descmod.EmitterData(nozzle,'VECTOR',2.,(0.,0.,-1.),80.,1.,250.),),particle_budget=1000)
        live=Solver.from_scene(desc,.04,substeps=2,device='cuda:0')
        live_output=wp.empty(live.num_particles,dtype=wp.vec3,device='cuda:0')
        for _ in range(6):live.step(live_output,use_graph=True)
        before=digest(live)
        replay=module.SolverReplay.clone_current(wp,live,expected_type=Solver,max_clone_bytes=256<<20)
        check('all_mutable_arrays_distinct',all(value.ptr!=getattr(live,name).ptr for name,value in vars(replay.solver).items() if isinstance(value,wp.array) and value.ptr))
        check('grid_and_graph_independent',replay.solver.grid.id!=live.grid.id and replay.solver.cuda_graph is None)
        check('collider_topology_retained_immutable',replay.solver.colliders is live.colliders)
        check('initial_state_exact',all(hashlib.sha256(getattr(replay.solver,name).numpy().tobytes()).hexdigest()==value for name,value in before['arrays'].items()))
        if inject_after_submit:
            original=replay.solver.step
            def fail_after_submit(*args,**kwargs):
                original(*args,**kwargs)
                raise RuntimeError('intentional failure after real CUDA work was submitted')
            replay.solver.step=fail_after_submit
            try:replay.advance_steps(1,max_steps=1)
            except RuntimeError:pass
            else:raise AssertionError('failure swallowed')
            check('failed_work_retained',replay.close()=='quarantined' and replay in module.process_state().quarantine and replay.solver is not None)
            retained=module.process_state();importlib.reload(module)
            check('reload_does_not_clear_quarantine',module.process_state() is retained and bool(retained.restart_required))
            report['status']='passed';report['requires_process_restart']=True
            return report
        reference=module.SolverReplay.clone_current(wp,live,expected_type=Solver,max_clone_bytes=256<<20)
        target=replay.initial_time_s+2.0/24.0
        calls=0
        while not replay.advance_to(target,max_steps=8):calls+=1
        replay.synchronize()
        # Independent direct calls to the actual solver at the same substeps,
        # not the replay controller again, establish numeric equivalence.
        with wp.ScopedStream(reference._stream):
            reference.solver.substeps=1
            while target-reference.solver.sim_time>1e-12:
                reference.solver.dt=min(reference.base_dt,target-reference.solver.sim_time)
                reference.solver.step(reference.output,use_graph=False)
        reference.synchronize()
        check('actual_solver_sample_time',abs(replay.solver.sim_time-target)<=1e-12,target=target,actual=replay.solver.sim_time,timer_slices=calls+1)
        for name in ('pos_sim','vel','pid','density_sorted','active'):
            np.testing.assert_array_equal(getattr(replay.solver,name).numpy(),getattr(reference.solver,name).numpy())
        check('matches_independent_direct_step_reference',True)
        check('emitter_state_advanced_only_in_clone',replay.solver._emitters[0].emitted>live._emitters[0].emitted,
              live_emitted=live._emitters[0].emitted,clone_emitted=replay.solver._emitters[0].emitted)
        check('source_unchanged_after_success',digest(live)==before)
        check('clone_settings_restored',replay.solver.dt==live.dt and replay.solver.substeps==live.substeps)
        check('success_cleanup',replay.close()==reference.close()=='completed')
        replay=reference=None
        cancelled=module.SolverReplay.clone_current(wp,live,expected_type=Solver,max_clone_bytes=256<<20)
        cancelled.advance_steps(3,max_steps=3)
        check('cancel_cleanup',cancelled.close()==cancelled.close()=='completed')
        cancelled=None
        check('source_unchanged_after_cancel',digest(live)==before)
        try:module.SolverReplay.clone_current(wp,live,expected_type=Solver,max_clone_bytes=1)
        except ValueError:pass
        else:raise AssertionError('clone byte budget ignored')
        check('source_unchanged_after_rejected_allocation',digest(live)==before)
        check('timeline_and_unrelated_native_data_preserved',bpy.context.scene.frame_current==original_frame and scene_sizes==(len(bpy.data.objects),len(bpy.data.meshes),len(bpy.data.node_groups)))
        report['live_before']=before;report['live_after']=digest(live)
        report['clone_memory_admission']='conservative per-module estimate; not total process VRAM'
        report['status']='passed'
        return report
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();raise
    finally:
        if not inject_after_submit:
            for owner in (replay,reference,cancelled):
                if owner is not None:owner.close()
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2),encoding='utf-8')
