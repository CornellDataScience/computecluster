# Architecture
## Head Node
Description: Raspberry Pi 4
Version: Ubuntu 22.04.5 LTS (jammy)
Ethernet IP: `10.0.0.1`
### CPU Information:
```
Architecture:                aarch64
  CPU op-mode(s):            32-bit, 64-bit
  Byte Order:                Little Endian
CPU(s):                      4
  On-line CPU(s) list:       0-3
Vendor ID:                   ARM
  Model name:                Cortex-A72
    Model:                   3
    Thread(s) per core:      1
    Core(s) per cluster:     4
    Socket(s):               -
    Cluster(s):              1
    Stepping:                r0p3
    CPU max MHz:             1800.0000
    CPU min MHz:             600.0000
    BogoMIPS:                108.00
    Flags:                   fp asimd evtstrm crc32 cpuid
Caches (sum of all):         
  L1d:                       128 KiB (4 instances)
  L1i:                       192 KiB (4 instances)
  L2:                        1 MiB (1 instance)
```
### GPU Information
N/A

## Compute1
Description: Large PC with glass door
Version: Ubuntu 24.04.3 LTS (noble)
Ethernet IP: `10.0.0.2`
### CPU Information
```
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          43 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   32
  On-line CPU(s) list:    0-31
Vendor ID:                AuthenticAMD
  Model name:             AMD Ryzen Threadripper 1950X 16-Core Processor
    CPU family:           23
    Model:                1
    Thread(s) per core:   2
    Core(s) per socket:   16
    Socket(s):            1
    Stepping:             1
    Frequency boost:      enabled
    CPU(s) scaling MHz:   68%
    CPU max MHz:          3400.0000
    CPU min MHz:          2200.0000
    BogoMIPS:             6786.60
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov 
                          pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_
                          opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc c
                          puid extd_apicid amd_dcm aperfmperf rapl pni pclmulqdq monitor
                           ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c 
                          rdrand lahf_lm cmp_legacy extapic cr8_legacy abm sse4a misalig
                          nsse 3dnowprefetch osvw skinit wdt tce topoext perfctr_core pe
                          rfctr_nb bpext perfctr_llc mwaitx cpb hw_pstate ssbd vmmcall f
                          sgsbase bmi1 avx2 smep bmi2 rdseed adx smap clflushopt sha_ni 
                          xsaveopt xsavec xgetbv1 clzero irperf xsaveerptr arat npt lbrv
                           svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeass
                          ists pausefilter pfthreshold avic v_vmsave_vmload vgif overflo
                          w_recov succor smca sev
Caches (sum of all):      
  L1d:                    512 KiB (16 instances)
  L1i:                    1 MiB (16 instances)
  L2:                     8 MiB (16 instances)
  L3:                     32 MiB (4 instances)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-31
```
### GPU Information
2x NVIDIA GeForce RTX 2080 Ti with 12GB RAM
Drivers installed

## Compute2
Description: Dell Lab PC
Version: Ubuntu 24.04.3 LTS (noble)
Ethernet IP: `10.0.0.3`
### CPU Information
```
Architecture:             x86_64
  CPU op-mode(s):         32-bit, 64-bit
  Address sizes:          46 bits physical, 48 bits virtual
  Byte Order:             Little Endian
CPU(s):                   20
  On-line CPU(s) list:    0-19
Vendor ID:                GenuineIntel
  Model name:             12th Gen Intel(R) Core(TM) i7-12700T
    CPU family:           6
    Model:                151
    Thread(s) per core:   2
    Core(s) per socket:   12
    Socket(s):            1
    Stepping:             2
    CPU(s) scaling MHz:   18%
    CPU max MHz:          4700.0000
    CPU min MHz:          800.0000
    BogoMIPS:             2764.80
    Flags:                fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov 
                          pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe sysc
                          all nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bt
                          s rep_good nopl xtopology nonstop_tsc cpuid aperfmperf tsc_kno
                          wn_freq pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ss
                          se3 sdbg fma cx16 xtpr pdcm sse4_1 sse4_2 x2apic movbe popcnt 
                          tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnow
                          prefetch cpuid_fault epb ssbd ibrs ibpb stibp ibrs_enhanced tp
                          r_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1
                           avx2 smep bmi2 erms invpcid rdseed adx smap clflushopt clwb i
                          ntel_pt sha_ni xsaveopt xsavec xgetbv1 xsaves split_lock_detec
                          t user_shstk avx_vnni dtherm ida arat pln pts hwp hwp_notify h
                          wp_act_window hwp_epp hwp_pkg_req hfi vnmi umip pku ospke wait
                          pkg gfni vaes vpclmulqdq tme rdpid movdiri movdir64b fsrm md_c
                          lear serialize pconfig arch_lbr ibt flush_l1d arch_capabilitie
                          s
Virtualization features:  
  Virtualization:         VT-x
Caches (sum of all):      
  L1d:                    512 KiB (12 instances)
  L1i:                    512 KiB (12 instances)
  L2:                     12 MiB (9 instances)
  L3:                     25 MiB (1 instance)
NUMA:                     
  NUMA node(s):           1
  NUMA node0 CPU(s):      0-19
```
### GPU Information
TODO
## Compute3
Description: Raspberry Pi 5
Version: Ubuntu 24.04.3 LTS (noble)
Ethernet IP: `10.0.0.4`
### CPU Information
```
Architecture:             aarch64
  CPU op-mode(s):         32-bit, 64-bit
  Byte Order:             Little Endian
CPU(s):                   4
  On-line CPU(s) list:    0-3
Vendor ID:                ARM
  Model name:             Cortex-A76
    Model:                1
    Thread(s) per core:   1
    Core(s) per cluster:  4
    Socket(s):            -
    Cluster(s):           1
    Stepping:             r4p1
    CPU(s) scaling MHz:   62%
    CPU max MHz:          2400.0000
    CPU min MHz:          1500.0000
    BogoMIPS:             108.00
    Flags:                fp asimd evtstrm aes pmull sha1 sha2 crc32 atomics fphp asimdh
                          p cpuid asimdrdm lrcpc dcpop asimddp
```
### GPU Information
N/A