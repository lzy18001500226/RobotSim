"""Exact-ABI observer I/O regression with a counter stub, never physics."""
from pathlib import Path
import argparse,hashlib,json,subprocess,time
REPO=Path(__file__).resolve().parents[4]
PREFIX=r'''#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
static FILE* delayed_stream;
static int delayed_writes;
static size_t slow_trace_write(const void*,size_t,size_t,FILE*);
#define fwrite slow_trace_write
'''
SUFFIX=r'''
#undef fwrite
static size_t slow_trace_write(const void* b,size_t s,size_t n,FILE* f) {
  if(f==delayed_stream && ++delayed_writes==10)usleep(350000);
  return fwrite(b,s,n,f);
}
static int stub_calls;
static void stub_step(const mjModel* m,mjData* d){(void)m;stub_calls++;d->time+=.025;}
static int stub_version(void){return 337;}
static int stub_id(const mjModel* m,int k,const char* n){(void)m;(void)k;(void)n;return -1;}
static const char* stub_name(const mjModel* m,int k,int i){(void)m;(void)k;(void)i;return NULL;}
static void already_initialized(void){}
int main(int argc,char** argv) {
  if(argc!=3)return 2;
  pthread_once(&once,already_initialized);
  step_real=stub_step;version_real=stub_version;id_real=stub_id;name_real=stub_name;
  char path[4096];snprintf(path,sizeof(path),"%s/trace.jsonl",argv[1]);
  sink=fopen(path,"w+");if(!sink)return 3;delayed_stream=sink;
  snprintf(latest_path,sizeof(latest_path),"%s/native_latest.json",argv[1]);
  snprintf(latest_tmp,sizeof(latest_tmp),"%s.tmp",latest_path);
  START_WRITER
  mjModel m={0};mjData d={0};m.opt.timestep=.001;
  int burst=atoi(argv[2]),calls=burst?2000:120;
  for(int i=0;i<calls;i++){mj_step(&m,&d);if(!burst)usleep(20000);}
  FINISH_WRITER
  rewind(sink);int c,records=0;
  while((c=fgetc(sink))!=EOF)if(c=='\n')records++;
  /* Count via the independent Python JSON parser too. */
  printf("{\"stub_step_calls\":%d,\"physics_dynamics_calls\":0,\"captured\":%d,\"written\":%d",stub_calls,count,records);
  EXTRA_STATUS
  puts("}");
  fclose(sink);sink=NULL;return 0;
}
'''
def run_case(source,folder,headers,*,asynchronous,burst=False):
    folder.mkdir()
    suffix=SUFFIX.replace('START_WRITER','if(trace_writer_start(sink))return 4;' if asynchronous else '')
    suffix=suffix.replace('FINISH_WRITER','trace_writer_finish();' if asynchronous else '')
    extra='printf('+json.dumps(',"queue_failed":%d,"rejected":%lld,"peak":%d')+',trace_failed,trace_rejected,trace_peak);'
    suffix=suffix.replace('EXTRA_STATUS',extra if asynchronous else '')
    c=folder/'probe.c';c.write_text(PREFIX+source+suffix)
    argv=['gcc','-O2','-Wall','-Wextra','-Werror','-I',str(headers),str(c),'-o',str(folder/'probe'),'-ldl','-pthread']
    build=subprocess.run(argv,capture_output=True,text=True)
    if build.returncode:raise RuntimeError(build.stderr)
    p=subprocess.Popen([str(folder/'probe'),str(folder),'1' if burst else '0'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    observed=[];stamp=None
    while p.poll() is None:
        file=folder/'native_latest.json'
        if file.exists():
            try:
                record=json.loads(file.read_text());value=record['host_monotonic_ns']
                if value!=stamp:observed.append(value);stamp=value
            except (OSError,ValueError):pass
        time.sleep(.003)
    out,err=p.communicate()
    rows=[json.loads(l) for l in (folder/'trace.jsonl').read_text().splitlines()]
    result=json.loads(out)
    result.update(returncode=p.returncode,stderr=err,records=len(rows),latest_unique_samples=len(observed),
        maximum_latest_capture_gap_s=max(((b-a)*1e-9 for a,b in zip(observed,observed[1:])),default=0.),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),compiler_command=argv)
    (folder/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--headers',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise ValueError('new output required')
    args.output.mkdir()
    path='scripts/research/issue46/native_runtime/native_grasp_audit.c'
    before=subprocess.check_output(['git','show','HEAD:'+path],cwd=REPO,text=True)
    after=(REPO/path).read_text()
    a=run_case(before,args.output/'synchronous',args.headers,asynchronous=False)
    b=run_case(after,args.output/'asynchronous',args.headers,asynchronous=True)
    c=run_case(after,args.output/'overflow',args.headers,asynchronous=True,burst=True)
    passed=(a['records']==b['records']==120 and a['stub_step_calls']==b['stub_step_calls']==120
            and a['maximum_latest_capture_gap_s']>.30 and b['maximum_latest_capture_gap_s']<.20
            and c.get('queue_failed')==1 and c.get('rejected',0)>0 and c['stub_step_calls']==2000)
    result={'classification':'C_INTERPOSITION_COUNTER_STUB_IO_FAULT_INJECTION_NO_PHYSICS','pass':passed,'before':a,'after':b,'overflow':c}
    (args.output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
    if not passed:raise SystemExit(1)
if __name__=='__main__':main()
