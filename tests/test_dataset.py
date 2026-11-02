
from src.small_visual_world.generate import generate
from src.small_visual_world.audit import audit
from src.small_visual_world.gallery import make_gallery
from src.small_visual_world.world import from_dict
from src.small_visual_world.render import render

def test_small_end_to_end_and_reproducibility(tmp_path):
    a=tmp_path/'one';b=tmp_path/'two'
    opts={'seed':47,'train_per_level':24,'eval_per_level':12,'adapt_train':12,'adapt_test':12}
    ma=generate(a,**opts);mb=generate(b,**opts)
    assert ma['manifests_sha256']==mb['manifests_sha256']
    assert ma['unique_scene_graphs']==mb['unique_scene_graphs']
    g=make_gallery(a)
    ra=audit(a)
    assert ra['pass']
    assert all(all(abs(v-0.5)<1e-9 for v in scores.values()) for scores in ra['non_neural_diagnostics']['exact_query_majority_accuracy'].values())
    assert all(v=='PASS' for v in ra['checks'].values())
    assert g['scenes']==24 and g['counterfactual_pairs']==8
    assert (a/'gallery_24_scenes.png').exists()
    assert (a/'counterfactuals_8_pairs.png').exists()
