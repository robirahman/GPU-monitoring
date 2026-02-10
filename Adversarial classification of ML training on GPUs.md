### Summary

Classifying workloads running on processors: Can we distinguish ML training from other workloads such as scientific computing, crypto mining, etc? What if the user is adversarially disguising their workload?

### Description of the project

Past research has proposed various ways of monitoring GPUs or clusters to detect when they are being used for ML workloads, or to measure the amount of ML training done. You can monitor various aspects of compute usage: network I/O memory usage and access patterns calculation format matrix/tensor operations instruction mix parallelism In summary, by using on-chip sensors measuring compute, memory, and instruction usage patterns, one can often classify the workload running on a GPU.

For instance, one proposal lists “memory transfer volume, off-chip transfer volume, arithmetic operation count, total instruction count, and tensor core usage” as useful properties for AI workload monitoring.

Even without privileged access to performance indicators, physical measurements can offer clues about a GPU’s workload. Every computation draws power and produces heat; the pattern of power draw over time or the chip’s thermal behavior can leak information about what type of program is running. These side-channel methods have the advantage of not needing to access any user data or internal states – they only observe external characteristics like power draw or EM emissions.

Using on-chip and side-channel sensors for workload classification is technically feasible – indeed, cloud providers already monitor GPU utilization, and tools exist to log power, temperature, etc., in real time. The difficulty lies in making this monitoring reliable and tamper-proof. A savvy user might deliberately modulate their workload to confuse such detectors (e.g. inserting dummy computations or idling periodically to mask a telltale pattern). They might also attempt to disable monitoring tools. Physical side channels could be mitigated by improved cooling or power smoothing to remove fluctuations. Moreover, these methods must avoid false positives.

Open questions:

* What input/output and computational characteristics most clearly distinguish ML training workloads from other intensive workloads (e.g. graphics rendering, scientific HPC simulations, crypto mining)?  
* Which network traffic patterns are uniquely indicative of distributed ML training?  
* Which hardware performance counters or on-chip sensors can most reliably classify the type of workload running?  
* What physical side-channel measurements are viable for distinguishing ML training, and are they robust across environments?  
* Can developers successfully disguise workloads such that we don’t know they’re doing AI computations on a monitored chip?  
* A determined actor might attempt to mask ML training by throttling usage, adding noise, faking counter readings, or modifying firmware.  
* Can we improve ML workload recognition to detect training runs that developers are attempting to hide?

Deliverables: Technical report detailing the project’s findings.

* Review of prior work  
* Comparison and contrast of characteristics of ML training vs other computational workloads  
* Results of any experiments or simulations conducted  
* Table or chart comparing monitorable signals, and summarizing their utility in workload discrimination  
* Proposal of a telemetry design for a HEM: recommendation of which metrics should be built into a hardware monitoring module so that it can catch unauthorized ML training Code for a prototype classifier that can distinguish ML training vs other usage as a function of signals and/or sensor data.

### Theory of change

The goal is to improve hardware governance by allowing regulations to target certain uses of computing hardware (e.g. frontier model training).

#### What role will mentees play in this project?

Mentees will conduct research that answers the open questions listed above, and possibly additional follow-up questions that we identify.

### Applicant prerequisites

* Ability to run a workload on a GPU. It's beneficial if you have your own. SPAR sometimes provides compute, but it’s unclear which usage information (such as power draw measurements) can be obtained from the cloud provider.

