import sys,tempfile,unittest
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ablation_experiments as exp
from ablation_models import configurations,dimensions,make_model,is_original,transformer_count,BASE_NAMES
from multi_dataset_fair_experiments import get_model_specs

class AblationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)
    def test_counts_and_budget(self):
        configs=configurations()
        self.assertEqual(len(configs),18)
        self.assertEqual(len({c['id'] for c in configs}),18)
        self.assertEqual(sum(is_original(c) for c in configs),3)
        for v in [65,75,91,88]:
            for c in configs[:12]:
                n=sum(p.numel() for p in make_model(c,v).parameters());d=dimensions(c,v)
                self.assertEqual(n,transformer_count(v,d['d_model'],d['ff_dim']))
                self.assertLess(abs(n/transformer_count(v,48,112)-1),.0012)
    def test_original_models_identical(self):
        for c in configurations():
            if is_original(c):
                torch.manual_seed(7);a=make_model(c,65)
                factory=next(f for n,_,f in get_model_specs() if n==BASE_NAMES[c['architecture']])
                torch.manual_seed(7);b=factory(65)
                self.assertEqual(exp.state_hash(a),exp.state_hash(b))
    def test_matched_initialization(self):
        for width in [48,32]:
            hashes=[]
            for c in configurations()[:12]:
                if c['width']==width:
                    torch.manual_seed(11);hashes.append(exp.state_hash(make_model(c,75)))
            self.assertEqual(len(set(hashes)),1)
    def test_training_defers_test_and_recovers_checkpoint(self):
        frozen=exp.plan();frozen['training_protocol']=dict(frozen['training_protocol'],max_epochs=1,steps_per_epoch=1)
        c=configurations()[0]
        with tempfile.TemporaryDirectory() as folder:
            a=exp.train_one('alice_in_wonderland',c,7,folder,frozen)
            self.assertNotIn('test_loss',a);self.assertNotIn('generation',a)
            p,cp=exp.record_paths(folder,'alice_in_wonderland',c,7);cp.unlink()
            b=exp.train_one('alice_in_wonderland',c,7,folder,frozen)
            self.assertEqual(a['best_val_loss'],b['best_val_loss'])
            self.assertTrue(cp.exists())
if __name__=='__main__':unittest.main()
