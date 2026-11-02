"""Generate the synthetic visual-question dataset, galleries, and integrity audit."""
from pathlib import Path
import json
from src.small_visual_world.generate import generate
from src.small_visual_world.gallery import make_gallery
from src.small_visual_world.audit import audit

PROJECT_ROOT=Path().cwd()
config=json.loads((PROJECT_ROOT/'configs'/'small_world'/'scene_generation.json').read_text())
folder=PROJECT_ROOT/'outputs'/'small_world'/'dataset'
metadata=generate(folder,**config)
make_gallery(folder)
report=audit(folder)
(folder/'audit_report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
print(f"Dataset audit: {'PASS' if report['pass'] else 'FAIL'} | {metadata['unique_scene_graphs']} disjoint scenes | {len(report['checks'])} checks")
if not report['pass']:raise SystemExit(1)
