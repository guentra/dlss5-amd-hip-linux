"""Compile the real trampoline into an ABI-aware CPU harness, never Wine."""
import os
import pathlib
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _tool(name):
    if name == 'x86_64-w64-mingw32-gcc' and os.environ.get('MINGW_CC'):
        path = pathlib.Path(os.environ['MINGW_CC']).expanduser()
        if path.is_file():
            return str(path)
    directories = []
    if os.environ.get('MINGW_PREFIX'):
        directories.append(pathlib.Path(os.environ['MINGW_PREFIX']).expanduser() / 'bin')
    if os.environ.get('TOOLCHAIN_DIR'):
        toolchain = pathlib.Path(os.environ['TOOLCHAIN_DIR']).expanduser()
        directories.append(toolchain / 'bin')
        directories.extend(sorted(toolchain.glob('llvm-mingw*/bin')))
    directories.extend(sorted((ROOT / 'toolchain').glob('llvm-mingw*/bin')))
    for directory in directories:
        candidate = directory / name
        if candidate.is_file():
            return str(candidate)
    found = shutil.which(name)
    if found:
        return found
    # Debian/Ubuntu's llvm packages install versioned names only
    # (llvm-readobj-21, never a bare llvm-readobj), so without this the
    # disassembly test below skips on every machine in that family even with
    # LLVM fully installed - a silent loss of coverage that looks like a
    # missing toolchain.
    versioned = sorted(pathlib.Path('/usr/bin').glob(name + '-*'))
    return str(versioned[-1]) if versioned else None


class TrampolineTests(unittest.TestCase):
    def test_pe_build_exports_and_sysv_registers(self):
        tools = {name: _tool(name) for name in
                 ('x86_64-w64-mingw32-gcc', 'llvm-readobj', 'llvm-objdump')}
        missing = [name for name, path in tools.items() if not path]
        if missing:
            self.skipTest('missing ' + ', '.join(missing) +
                          '; set MINGW_CC / MINGW_PREFIX / TOOLCHAIN_DIR or install mingw-w64 and llvm')
        with tempfile.TemporaryDirectory() as tmp:
            dll = pathlib.Path(tmp) / 'dlss5_hip.dll'
            def command(tool, *args):
                result = subprocess.run([tools[tool], *map(str, args)],
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                return result.stdout
            command('x86_64-w64-mingw32-gcc', '-shared', '-O2', '-Wall', '-Wextra', '-Werror',
                    '-o', dll, ROOT / 'hip/src/trampoline.c')
            exports = command('llvm-readobj', '--coff-exports', dll)
            self.assertEqual(set(re.findall(r'Name: (dlss5_\w+)', exports)),
                             {'dlss5_init', 'dlss5_run', 'dlss5_shutdown', 'dlss5_last_error',
                              'dlss5_run_frame', 'dlss5_find_device', 'dlss5_run_frame_raw_gpu'})
            imports = command('llvm-readobj', '--coff-imports', dll)
            self.assertIn('GetProcAddress', imports)
            for forbidden in ('CreateFile', 'ReadFile', 'WriteFile', 'GetEnvironmentVariable'):
                self.assertNotIn(forbidden, imports)
            self.assertIn(b'__wine_get_unix_env', dll.read_bytes())
            self.assertNotIn(b'dlss5_hip_bridge.addr', dll.read_bytes())
            disasm = command('llvm-objdump', '-d', '--disassemble-symbols=dlss5_run', dll)
            self.assertRegex(disasm, r'movq\s+%rcx, %rdi')
            self.assertRegex(disasm, r'movq\s+%rdx, %rsi')
            # llvm-mingw moves r8d straight into edx; gcc may spill then reload.
            self.assertRegex(disasm, r'movl\s+%r8d,')
            self.assertRegex(disasm, r'movl\s+\S+, %edx')
            # TLS diagnostics can change the allocated call-target register;
            # the argument ABI above, not a compiler's %rax choice, is fixed.
            self.assertRegex(disasm, r'callq\s+\*%r(?:[abcd]x|[89]|1[01])\b')

    def test_built_dll_has_no_toolchain_runtime_imports(self):
        """The trampoline must load with nothing but the game beside it.

        Built by a GCC-flavoured mingw (Debian/Ubuntu's gcc-mingw-w64-x86-64,
        Fedora's mingw-w64-gcc, what hip/Makefile's discovery finds when no
        llvm-mingw drop is installed) the default link pulls in
        libgcc_s_seh-1.dll and libwinpthread-1.dll, which the game's
        directory does not have. LoadLibrary then fails and the add-on
        reports "dlss5_hip.dll missing (HIP trampoline)" - indistinguishable
        from the file being absent, while it sits right there. Found by
        deploying such a build and watching every frame fall back to the
        original image."""
        gcc = _tool('x86_64-w64-mingw32-gcc')
        if not gcc:
            self.skipTest('no mingw gcc available to build the trampoline')
        objdump = shutil.which('objdump') or _tool('x86_64-w64-mingw32-objdump')
        if not objdump:
            self.skipTest('no objdump available to read the import table')
        with tempfile.TemporaryDirectory() as tmp:
            dll = pathlib.Path(tmp) / 'dlss5_hip.dll'
            # -static matches hip/Makefile exactly: on some distros
            # (Fedora's mingw-w64-gcc) -static-libgcc alone still imports
            # libwinpthread-1.dll, so the test must verify the real shipped
            # link line, not a weaker variant of it.
            build = subprocess.run([str(gcc), '-shared', '-O2', '-static',
                                    '-o', str(dll), str(ROOT / 'hip' / 'src' / 'trampoline.c')],
                                   text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            # Pin the locale: a French-locale binutils objdump prints
            # "Nom de la DLL" instead of "DLL Name", which breaks the parse.
            dump = subprocess.run([str(objdump), '-p', str(dll)], text=True,
                                  capture_output=True,
                                  env=dict(os.environ, LC_ALL='C'))
            self.assertEqual(dump.returncode, 0, dump.stderr)
            imports = re.findall(r'DLL Name:\s*(\S+)', dump.stdout)
            self.assertTrue(imports, dump.stdout)
            # Every import has to be something Windows (or Wine) already
            # provides, never a file the toolchain expects shipped alongside.
            shipped = [name for name in imports
                       if name.lower().startswith(('libgcc', 'libwinpthread', 'libstdc++'))]
            self.assertEqual(shipped, [],
                             'trampoline imports toolchain runtime DLLs: ' + ', '.join(imports))

    def test_live_unix_environment_only_and_forwarding(self):
        source = r'''
#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include "hip/src/trampoline.c"
_Static_assert(sizeof(Dlss5HipBridge)==40,"bridge size ABI");
_Static_assert(offsetof(Dlss5HipBridge,init)==8,"init ABI");
_Static_assert(offsetof(Dlss5HipBridge,run)==16,"run ABI");
_Static_assert(offsetof(Dlss5HipBridge,shutdown)==24,"shutdown ABI");
_Static_assert(offsetof(Dlss5HipBridge,last_error)==32,"error ABI");
static char live[80],pe[80];
static int env_available=1,env_status=0,files=0,pe_reads=0,stops=0;
static int host_init(const char *p,int gpu){assert(!strcmp(p,"weights"));return gpu+10;}
static int host_run(const float *p,float *out,unsigned seed){out[0]=p[0]+seed;return 7;}
static void host_stop(void){stops++;}
static const char *host_error(void){return "host diagnostic";}
static Dlss5HipBridge table={DLSS5_HIP_MAGIC,host_init,host_run,host_stop,host_error};
static int frame_calls;
static int host_frame(const Dlss5Frame *f){assert(f->struct_size==sizeof(*f));assert(f->seed==27);++frame_calls;return 11;}
static Dlss5HipFrameBridge frame_table={DLSS5_HIP_FRAME_MAGIC,host_frame};
static int host_device(const char *name){assert(!strcmp(name,"exact adapter"));return 7;}
static Dlss5HipDeviceBridge device_table={DLSS5_HIP_DEVICE_MAGIC,host_device};
static int __stdcall unix_env(const char *name,char *buf,unsigned long long n){
 if(!strcmp(name,DLSS5_HIP_DEVICE_ENV)){snprintf(buf,(size_t)n,"%p",(void*)&device_table);return 0;}
 if(!strcmp(name,DLSS5_HIP_FRAME_ENV)){snprintf(buf,(size_t)n,"%p",(void*)&frame_table);return 0;}
 assert(!strcmp(name,DLSS5_HIP_ENV));
 if(env_status)return env_status;
 if(strlen(live)>=n){memset(buf,'a',(size_t)n);return 0;}
 strcpy(buf,live);return 0;
}
HANDLE __stdcall GetModuleHandleA(const char *name){assert(!strcmp(name,"ntdll.dll"));return (HANDLE)1;}
void *__stdcall GetProcAddress(HANDLE m,const char *name){assert(m);assert(!strcmp(name,"__wine_get_unix_env"));return env_available?(void*)unix_env:0;}
DWORD __stdcall GetEnvironmentVariableA(const char *name,char *buf,DWORD n){(void)name;pe_reads++;if(strlen(pe)<n)strcpy(buf,pe);return (DWORD)strlen(pe);}
HANDLE __stdcall CreateFileA(const char*n,DWORD a,DWORD s,void*x,DWORD d,DWORD f,HANDLE t){(void)n;(void)a;(void)s;(void)x;(void)d;(void)f;(void)t;files++;return (HANDLE)(long long)-1;}
BOOL __stdcall WriteFile(HANDLE h,const void*b,DWORD n,DWORD*w,void*o){(void)h;(void)b;(void)n;(void)w;(void)o;return 0;}
BOOL __stdcall ReadFile(HANDLE h,void*b,DWORD n,DWORD*w,void*o){(void)h;(void)b;(void)n;(void)w;(void)o;return 0;}
BOOL __stdcall CloseHandle(HANDLE h){(void)h;return 1;}
int main(void){
 snprintf(pe,sizeof pe,"%p",(void*)&table);
 // A valid-looking inherited PE pointer is NOT an allowed fallback.
 DllMain(0,DLL_PROCESS_ATTACH,0);
 assert(dlss5_init("weights",2)==-1);
 assert(pe_reads==0 && files==0);
 const char *bad[]={"","0x","0","xyz","0x123junk","0x10000000000000000"};
 for(unsigned i=0;i<sizeof bad/sizeof bad[0];i++){
  strcpy(live,bad[i]);g=0;init_bridge();assert(dlss5_run(0,0,0)==-1);
 }
 memset(live,'a',sizeof live-1);live[sizeof live-1]=0;
 g=0;init_bridge();assert(dlss5_init("weights",2)==-1);
 strcpy(live,pe);env_status=-1;g=0;init_bridge();assert(dlss5_init("weights",2)==-1);
 env_status=0;env_available=0;g=0;init_bridge();assert(dlss5_init("weights",2)==-1);
 env_available=1;g=0;init_bridge();
 assert(dlss5_init("weights",2)==12);
 assert(dlss5_find_device("exact adapter")==7);
 float in=3,out=0;assert(dlss5_run(&in,&out,4)==7 && out==7);
 assert(!strcmp(dlss5_last_error(),"host diagnostic"));
 Dlss5Frame frame={0};frame.struct_size=sizeof(frame);frame.seed=27;
 assert(dlss5_run_frame(&frame)==11 && frame_calls==1);
 frame_table.magic=0;assert(dlss5_run_frame(&frame)==-1);
 assert(strstr(dlss5_last_error(),"frame bridge missing"));
 frame_table.magic=DLSS5_HIP_FRAME_MAGIC;
 assert(dlss5_run_frame(&frame)==11 && frame_calls==2);
 assert(!strcmp(dlss5_last_error(),"host diagnostic"));
 dlss5_shutdown();assert(stops==1);
 table.magic=0;assert(dlss5_init("weights",2)==-1);table.magic=DLSS5_HIP_MAGIC;
 table.run=0;assert(dlss5_run(&in,&out,4)==-1);
 assert(files==0 && pe_reads==0);
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            src = pathlib.Path(tmp) / 'test.c'
            src.write_text(source)
            exe = pathlib.Path(tmp) / 'test'
            build = subprocess.run(['cc', '-std=c11', '-O2', '-D__declspec(x)=',
                                    '-D__stdcall=__attribute__((ms_abi))',
                                    '-I', str(ROOT), str(src), '-o', str(exe)],
                                   text=True, capture_output=True)
            self.assertEqual(build.returncode, 0, build.stdout + build.stderr)
            run = subprocess.run([str(exe)], text=True, capture_output=True)
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == '__main__':
    unittest.main()
