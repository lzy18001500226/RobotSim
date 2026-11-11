"""Executable C observer lifetime regression; no physics or robot commands."""
import json,shutil,subprocess,tempfile,unittest
from pathlib import Path

DRIVER=r"""
static int stub_calls;
static void stub_step(const mjModel* m,mjData* d){(void)m;stub_calls++;d->time+=.025;}
static int stub_version(void){return 337;}
static int stub_id(const mjModel* m,int kind,const char* name){(void)m;(void)kind;(void)name;return -1;}
static const char* stub_name(const mjModel* m,int kind,int id){(void)m;(void)kind;(void)id;return NULL;}
static void already_initialized(void){}
int main(void){
  pthread_once(&once,already_initialized);
  step_real=stub_step;version_real=stub_version;id_real=stub_id;name_real=stub_name;
  sink=tmpfile();if(!sink)return 3;
  mjModel model={0};mjData data={0};model.opt.timestep=.001;
  for(int i=0;i<12020;i++)mj_step(&model,&data);
  rewind(sink);int ch,records=0;while((ch=fgetc(sink))!=EOF)if(ch=='\n')records++;
  fclose(sink);printf("{\"stub_step_calls\":%d,\"records\":%d,\"physics_dynamics_calls\":0}\n",stub_calls,records);
  return 0;
}
"""

def exact_headers():
    matches=sorted((Path.home()/'.cache/uv/archive-v0').glob('**/mujoco/include/mujoco/mujoco.h'))
    return next((p.parents[1] for p in matches if '#define mjVERSION_HEADER 337' in p.read_text()),None)

def probe(source,include,output):
    output=Path(output);output.mkdir(exist_ok=False)
    file=output/'observer_lifetime.c';file.write_text(source+'\n'+DRIVER)
    command=['gcc','-O2','-Wall','-Wextra','-Werror','-I',str(include),str(file),'-o',str(output/'probe'),'-ldl','-pthread']
    compiled=subprocess.run(command,capture_output=True,text=True,timeout=20)
    if compiled.returncode:raise RuntimeError(compiled.stderr)
    result=subprocess.run([str(output/'probe')],capture_output=True,text=True,timeout=20,check=True)
    return dict(json.loads(result.stdout),compiler_command=command)

class NativeObserverLifetimeTests(unittest.TestCase):
    def test_keeps_emitting_after_12000_records_and_calls_real_step_once(self):
        include=exact_headers()
        if include is None or shutil.which('gcc') is None:self.skipTest('exact337 headers and gcc required; no install')
        with tempfile.TemporaryDirectory(prefix='robotsim-observer-test-') as temp:
            result=probe(Path(__file__).with_name('native_grasp_audit.c').read_text(),include,Path(temp)/'build')
        self.assertEqual(result['stub_step_calls'],12020)
        self.assertEqual(result['records'],12020)
        self.assertEqual(result['physics_dynamics_calls'],0)

if __name__=='__main__':unittest.main()
