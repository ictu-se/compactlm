import sys,unittest,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corrected_experiments import batch,evaluate,generate,generate_stateful,checkpoint_decision,CONFIG,load_corpus,train_job,code_hash
from multi_dataset_fair_experiments import get_model_specs
class ProtocolTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(1)
 def test_boundary_window(self):
  x,y=batch(torch.arange(5),4,3,torch.Generator().manual_seed(1))
  self.assertEqual(x.tolist(),[[0,1,2,3]]*3);self.assertEqual(y.tolist(),[[1,2,3,4]]*3)
 def test_fixed_evaluation(self):
  m=get_model_specs()[0][2](7);s=torch.arange(132)%7
  state=torch.random.get_rng_state().clone();a,n=evaluate(m,s);b,k=evaluate(m,s)
  self.assertEqual(n,131);self.assertEqual(a,b);self.assertTrue(torch.equal(state,torch.random.get_rng_state()))
 def test_checkpoint_and_patience_independent(self):
  b,p,n,save=checkpoint_decision(.999,1.,1.,0,CONFIG)
  self.assertTrue(save);self.assertEqual(b,.999);self.assertEqual(p,1.);self.assertEqual(n,1)
 def test_causality_and_context(self):
  for name,_,factory in get_model_specs():
   m=factory(7).eval();x=torch.randint(7,(2,12));y=x.clone();y[:,7:]=(y[:,7:]+1)%7
   self.assertTrue(torch.allclose(m(x)[:,:7],m(y)[:,:7],atol=1e-6),name)
   lens=[];h=m.register_forward_pre_hook(lambda _,a:lens.append(a[0].shape[1]))
   generate(m,torch.randint(7,(2,96)),4);h.remove();self.assertEqual(lens,[64]*4,name)
 def test_stateful_equivalence(self):
  for name,_,factory in get_model_specs()[:-1]:
   m=factory(7).eval();p=torch.randint(7,(2,8))
   a=generate_stateful(m,p,6);b=generate(m,p,6,context=100)
   self.assertTrue(torch.equal(a,b),name)
 def test_seed_independence(self):
  batches=[]
  for _,_,factory in get_model_specs():
   torch.manual_seed(7);factory(7);batches.append(batch(torch.arange(200),64,4,torch.Generator().manual_seed(107))[0])
  self.assertTrue(all(torch.equal(batches[0],b) for b in batches))
 def test_fresh_run_and_missing_checkpoint(self):
  cfg=dict(CONFIG,max_epochs=1,steps_per_epoch=1,prompts=2,continuation=4)
  with tempfile.TemporaryDirectory() as d:
   name=get_model_specs()[0][0];a=train_job('alice_in_wonderland',name,7,d,cfg,'test',False)
   p=Path(d)/'alice_in_wonderland'/a['checkpoint'];p.unlink()
   b=train_job('alice_in_wonderland',name,7,d,cfg,'test',True)
   self.assertTrue(p.exists());self.assertEqual(a['final_test_loss'],b['final_test_loss'])
   self.assertEqual(b['evaluation_tokens'],b['corpus']['split_sizes'][2]-1)
if __name__=='__main__':unittest.main()
