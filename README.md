# Master-Thesis
My Computer Science Master thesis on optimising CUDA kernels for Tensor Network computations.

https://daniel-sinkin.github.io/master-thesis/

https://www.youtube.com/watch?v=QQceTDjA4f4
 * "Your blocks must never be less than 128 threads (4 warps) as they consecutively are able to load an entire memory page together (128 * 8 = 1024 Byte)"
 * Memory Patterns make up a difference of 10x therefore they are the most important aspect; occupancy is the second most important as it makes up a factor of 2.
 * THe more you can oversubscribe the GPU the better performance you will get, there are a lot of tricks the runtime can do to slot in things efficiently.
 * "Our primary limiting factor is memory bandwidth"

https://www.youtube.com/watch?v=GmNkYayuaA4

https://www.youtube.com/watch?v=-tIQbIhTAv8
