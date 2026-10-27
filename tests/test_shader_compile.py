"""Offline GLSL checks for every viewport shader (no GPU, no Blender).

The gpu-module shaders (SSFR passes, points/spheres) are bodies only: Blender generates their
declarations from the GPUShaderCreateInfo. Their create infos are recorded from the real
``_build`` code and turned into the equivalent GLSL 4.50 declarations, so a source that uses a
push constant, sampler or interface variable its create info does not declare fails here.

* A structural check always runs.
* A full compile + link runs through Khronos glslangValidator when it is available (PATH, or
  ``WARPSPH_GLSLANG``); otherwise those tests skip. This proves the GLSL is valid, not that a
  given driver or Blender's Vulkan/SPIR-V translation accepts it; that stays the live probe's job.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from warp_sph_realtime.render import gpu_points as gp
from warp_sph_realtime.render import gpu_ssfr as gs
from warp_sph_realtime.render import ssfr_shaders as legacy

GLSL_TYPES = {'FLOAT': 'float', 'VEC2': 'vec2', 'VEC3': 'vec3', 'VEC4': 'vec4', 'MAT4': 'mat4', 'INT': 'int',
              'FLOAT_2D': 'sampler2D'}


class _Iface:
    def __init__(self, name):
        self.name, self.items = name, []

    def smooth(self, kind, name):
        self.items.append(('smooth', kind, name))

    def flat(self, kind, name):
        self.items.append(('flat', kind, name))


class _Info:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return lambda *args: self.calls.append((name, *args))

    def of(self, kind):
        return [c[1:] for c in self.calls if c[0] == kind]

    @property
    def iface(self) -> _Iface:
        return self.of('vertex_out')[0][0]


def _recording_gpu():
    return SimpleNamespace(types=SimpleNamespace(GPUShaderCreateInfo=_Info, GPUStageInterfaceInfo=_Iface),
                           shader=SimpleNamespace(create_from_info=lambda info: info))


def gpu_module_shaders() -> dict:
    """name -> recorded create info, straight from the renderers' own build code."""
    ssfr = gs.SSFRRenderer(_recording_gpu(), lambda *a: None)
    ssfr._build()
    points = gp.PointRenderer(_recording_gpu(), lambda *a: None)
    points._build()
    return {**{f'ssfr_{name}': info for name, info in ssfr._shaders.items()}, 'points': points._shader}


def stage_sources(info: _Info) -> tuple[str, str]:
    """The create info as GLSL 4.50 declarations prepended to the vertex and fragment bodies."""
    common = ['#version 450 core']
    common += [f'uniform {GLSL_TYPES[kind]} {name};' for kind, name in info.of('push_constant')]
    common += [f'uniform {GLSL_TYPES[kind]} {name};' for _slot, kind, name in info.of('sampler')]
    vert = common + ['#define gpu_InstanceIndex gl_InstanceID']
    vert += [f'layout(location = {slot}) in {GLSL_TYPES[kind]} {name};' for slot, kind, name in info.of('vertex_in')]
    vert += [f'{qual} out {GLSL_TYPES[kind]} {name};' for qual, kind, name in info.iface.items]
    frag = common + [f'{qual} in {GLSL_TYPES[kind]} {name};' for qual, kind, name in info.iface.items]
    frag += [f'layout(location = {slot}) out {GLSL_TYPES[kind]} {name};' for slot, kind, name in info.of('fragment_out')]
    (vertex,), (fragment,) = info.of('vertex_source'), info.of('fragment_source')
    return '\n'.join(vert) + '\n' + vertex[0], '\n'.join(frag) + '\n' + fragment[0]


def legacy_programs() -> dict:
    """The raw-GL programs, paired exactly as viewport_draw links them."""
    from test_ssfr_renderer import _load_viewport_draw
    sphere = _load_viewport_draw().SPHERE_IMPOSTOR_FRAG
    return {
        'legacy_depth': (legacy.SSFR_DEPTH_VERT, legacy.SSFR_DEPTH_FRAG),
        'legacy_thickness': (legacy.SSFR_DEPTH_VERT, legacy.SSFR_THICKNESS_FRAG),
        'legacy_filter': (legacy.SSFR_FULLSCREEN_VERT, legacy.SSFR_NARROW_RANGE_FRAG),
        'legacy_composite': (legacy.SSFR_FULLSCREEN_VERT, legacy.SSFR_COMPOSITE_FRAG),
        'legacy_sphere': (legacy.SSFR_DEPTH_VERT, sphere),
    }


def all_programs() -> dict:
    return {**{name: stage_sources(info) for name, info in gpu_module_shaders().items()}, **legacy_programs()}


# -- structural (always runs) -----------------------------------------------------------------
@pytest.mark.parametrize('name', sorted(gpu_module_shaders()))
def test_sources_only_use_what_the_create_info_declares(name):
    info = gpu_module_shaders()[name]
    (vertex,), (fragment,) = info.of('vertex_source'), info.of('fragment_source')
    source = vertex[0] + fragment[0]
    push = {n for _k, n in info.of('push_constant')}
    samplers = {n for _s, _k, n in info.of('sampler')}
    assert set(re.findall(r'\bu_\w+', source)) <= push, 'push constant used but not declared'
    assert set(re.findall(r'\b(?:texelFetch|textureSize|texture)\(\s*(\w+)', source)) <= samplers
    for _q, _k, var in info.iface.items:
        assert re.search(rf'\b{var}\s*=', vertex[0]), f'interface {var} never written by the vertex stage'
    for _slot, _k, var in info.of('vertex_in'):
        assert re.search(rf'\b{var}\b', vertex[0]), f'vertex input {var} unused'


# -- full compile + link (glslangValidator) ---------------------------------------------------
def _validator() -> str | None:
    configured = os.environ.get('WARPSPH_GLSLANG')
    if configured:
        return configured
    return shutil.which('glslangValidator') or shutil.which('glslang')


@pytest.mark.parametrize('name', sorted(all_programs()))
def test_glsl_compiles_and_links(name, tmp_path: Path):
    tool = _validator()
    if not tool:
        pytest.skip('glslangValidator not found (set WARPSPH_GLSLANG or put it on PATH)')
    vertex, fragment = all_programs()[name]
    vert, frag = tmp_path / f'{name}.vert', tmp_path / f'{name}.frag'
    vert.write_text(vertex, encoding='utf-8')
    frag.write_text(fragment, encoding='utf-8')
    proc = subprocess.run([tool, '-l', str(vert), str(frag)], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f'{name}:\n{proc.stdout}\n{proc.stderr}'
