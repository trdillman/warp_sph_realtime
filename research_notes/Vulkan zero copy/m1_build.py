"""Validate/build the M1 extension artifact in isolated user directories; never install."""
import hashlib,json,os,subprocess,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vulkan-zero-copy/20261018-m1'
exe=json.loads((ROOT/'evidence/vulkan-zero-copy/20261017-m0/host-preflight.json').read_text())['candidate']['exe']
env=dict(os.environ)
for key in ('RESOURCES','CONFIG','SCRIPTS','DATAFILES','EXTENSIONS'):
    target=OUT/'build-profile'/key.lower();target.mkdir(parents=True,exist_ok=True)
    env['BLENDER_USER_'+key]=str(target)
artifact=OUT/'artifact';artifact.mkdir(exist_ok=True)
rows=[]
for name,args in [('validate',['validate',str(ROOT/'warp_sph_realtime')]),
                  ('build',['build','--source-dir',str(ROOT/'warp_sph_realtime'),'--output-dir',str(artifact)])]:
    command=[exe,'--factory-startup','--offline-mode','--command','extension',*args]
    result=subprocess.run(command,capture_output=True,text=True,env=env,timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    (OUT/f'package-{name}.log').write_text(result.stdout+'\n'+result.stderr)
    rows.append({'command':command,'exit_code':result.returncode})
    assert result.returncode==0,rows[-1]
archives=list(artifact.glob('*.zip'));assert len(archives)==1
archive=archives[0]
with zipfile.ZipFile(archive) as z:
    names=z.namelist()
    required=['render/blender_symbols.py','render/blender_symbols_helper.py',
              'render/profiles/blender-5.2.0-fbe6228777e7-windows-amd64-cp313-v1.json']
    for name in required:
        assert z.read(name)==(ROOT/'warp_sph_realtime'/name).read_bytes()
    assert not any(name.lower().endswith('.pdb') for name in names)
receipt={'checks':rows,'archive':str(archive.relative_to(ROOT)),
         'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'resolver_members_match':required,
         'installed':False,'isolated_user_paths':{k:v for k,v in env.items() if k.startswith('BLENDER_USER_')}}
(OUT/'package.json').write_text(json.dumps(receipt,indent=2))
print(json.dumps(receipt,indent=2))
