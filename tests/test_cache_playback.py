"""Playback adapter (P1.6): point cloud object, hold-nearest clamping, change detection and handler safety,
against a fake bpy in a separate interpreter (real Blender 5.2 is covered by the live check)."""
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r'''
import logging, sys, tempfile, types
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np

class Data:
    def __init__(self): self.values = {}
    def foreach_set(self, field, arr): self.values[field] = np.array(arr).copy()

class Attr:
    def __init__(self, name, data_type, domain): self.name, self.data_type, self.domain, self.data = name, data_type, domain, Data()

class Attrs(dict):
    def get(self, name, default=None): return dict.get(self, name, default)
    def new(self, name, data_type, domain):
        self[name] = Attr(name, data_type, domain); return self[name]
    def remove(self, attr): del self[attr.name]

class Cloud:
    users = 1
    def __init__(self):
        self.points, self.attributes, self.updates = [], Attrs(), 0
        self.attributes["position"] = Attr("position", "FLOAT_VECTOR", "POINT")
    def resize(self, n): self.points = [None] * n
    def as_pointer(self): return id(self)
    def update_tag(self): self.updates += 1

class Obj(dict):
    def __init__(self, name, data): dict.__init__(self); self.name, self.data, self.type = name, data, "POINTCLOUD"

objects = []
bpy = types.ModuleType("bpy")
bpy.data = NS(filepath="", pointclouds=NS(new=lambda name: Cloud(), remove=lambda c: None),
              objects=NS(new=lambda name, data: Obj(name, data), remove=lambda o: objects.remove(o)))
bpy.path = NS(abspath=lambda p: p)
def persistent(f): f._bpy_persistent = True; return f
bpy.app = NS(handlers=NS(frame_change_post=[], persistent=persistent))
sys.modules["bpy"] = bpy

from warp_sph_realtime.cache import blender_adapter as ba
from warp_sph_realtime.cache.disk import CacheError, DiskCache

tmp = tempfile.TemporaryDirectory()
folder = Path(tmp.name) / "cache"
scene = NS(name="Scene", frame_current=1, objects=objects, warp_sph=NS(cache_dir=str(folder), particle_radius=0.02),
           collection=NS(objects=NS(link=lambda o: objects.append(o))))
disk = DiskCache(folder)
disk.begin(10, {"particles": 5})

def frame(n, count=5):
    rng = np.random.default_rng(n)
    return (rng.standard_normal((count, 3)).astype(np.float32), rng.standard_normal((count, 3)).astype(np.float32),
            rng.permutation(count).astype(np.int32))

def write(n, count=5):
    disk.write_frame(n, *frame(n, count))

def loaded(cloud):
    return {name: attr.data.values for name, attr in cloud.attributes.items()}

assert ba.find_playback_object(scene) is None
assert ba.show_frame(scene, 10) is False                       # no object yet: nothing to do
obj = ba.ensure_playback_object(scene)
assert ba.ensure_playback_object(scene) is obj and len(objects) == 1 and obj[ba.PLAYBACK_PROP] is True
assert ba.find_playback_object(scene) is obj
try:
    ba.show_frame(scene, 10); raise SystemExit("an empty cache must raise")
except CacheError:
    pass

for n in range(10, 14): write(n)
assert ba.show_frame(scene, 12) is True
cloud = obj.data
pos, vel, ids = frame(12)
v = loaded(cloud)
assert len(cloud.points) == 5 and cloud.updates == 1
assert v["position"]["vector"].tobytes() == pos.ravel().tobytes()          # bit-identical to the file
assert v["velocity"]["vector"].tobytes() == vel.ravel().tobytes() and v["id"]["value"].tolist() == ids.tolist()
assert cloud.attributes["velocity"].data_type == "FLOAT_VECTOR" and cloud.attributes["id"].data_type == "INT"
assert cloud.attributes["radius"].data_type == "FLOAT" and v["radius"]["value"].tolist() == [np.float32(0.02)] * 5
print("exact_frame_ok")

assert ba.show_frame(scene, 12) is False and cloud.updates == 1             # same frame, same file: no reload
assert ba.show_frame(scene, 13) is True and cloud.updates == 2
assert ba.show_frame(scene, 99) is False                                     # past the end: nearest (13) is already shown
write(13)                                                                    # same bytes rewritten: mtime changes
import os, time
os.utime(disk.path(13), ns=(time.time_ns() + 10**9, time.time_ns() + 10**9))
assert ba.show_frame(scene, 13) is True                                      # a re-baked frame is picked up
assert ba.show_frame(scene, -50) is True and loaded(cloud)["position"]["vector"].tobytes() == frame(10)[0].ravel().tobytes()
assert ba.show_frame(scene, 500) is True and loaded(cloud)["position"]["vector"].tobytes() == frame(13)[0].ravel().tobytes()
print("clamp_and_change_ok")

# a gap ends the cached range: frames after it are not shown
write(15)
assert disk.cached_range() == (10, 13)
assert ba.show_frame(scene, 15) is False                                     # clamped to 13, already shown
print("range_ok")

# particle count change: points resized, radius rewritten to the new length
DiskCache(folder).clear(); disk.begin(10, {"particles": 8})
write(10, 8)
assert ba.show_frame(scene, 10) is True and len(cloud.points) == 8
assert len(loaded(cloud)["radius"]["value"]) == 8 and len(loaded(cloud)["id"]["value"]) == 8
scene.warp_sph.particle_radius = 0.05                                        # a radius change refreshes the attribute
assert ba.show_frame(scene, 10) is True and loaded(cloud)["radius"]["value"][0] == np.float32(0.05)
print("resize_ok")

# the handler never raises into Blender, and logs each failure once
records = []
class Capture(logging.Handler):
    def emit(self, record): records.append(record.getMessage())
logging.getLogger("warp_sph_realtime.cache.blender_adapter").addHandler(Capture())
DiskCache(folder).clear()
for _ in range(3):
    ba.on_frame_change_post(scene, None)
assert len(records) == 1 and "no cached frames" in records[0], records
bare = NS(name="Scene", frame_current=1, objects=[], warp_sph=scene.warp_sph)
ba.on_frame_change_post(bare)                                                # no playback object: silent no-op
ba.on_frame_change_post(None)                                                # garbage scene: still no exception
assert len(records) == 2
print("handler_ok")

handlers = bpy.app.handlers.frame_change_post
ba.register(); ba.register()
assert handlers == [ba.on_frame_change_post] and ba.on_frame_change_post._bpy_persistent
ba.unregister()
assert handlers == []
print("register_ok")

assert ba.remove_playback_object(scene) is True and objects == [] and ba.remove_playback_object(scene) is False
print("PLAYBACK_OK")
'''


class PlaybackAdapterTest(unittest.TestCase):
    def test_playback_adapter_with_fake_bpy(self):
        result = subprocess.run([sys.executable, '-'], input=SCRIPT, cwd=str(ROOT),
                                capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        for marker in ('exact_frame_ok', 'clamp_and_change_ok', 'range_ok', 'resize_ok', 'handler_ok', 'register_ok',
                       'PLAYBACK_OK'):
            self.assertIn(marker, result.stdout)


if __name__ == '__main__':
    unittest.main()
