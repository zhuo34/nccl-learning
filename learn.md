# NCCl 源码解析
基于 v2.22.3-1。

### `ncclGetUniqueId()` 函数

```cpp
NCCL_API(ncclResult_t, ncclGetUniqueId, ncclUniqueId* out);
ncclResult_t ncclGetUniqueId(ncclUniqueId* out) {
  NCCLCHECK(ncclInit());
  NCCLCHECK(PtrCheck(out, "GetUniqueId", "out"));
  ncclResult_t res = bootstrapGetUniqueId((struct ncclBootstrapHandle*)out);
  TRACE_CALL("ncclGetUniqueId(0x%llx)", (unsigned long long)hashUniqueId(*out));
  return res;
}
```

用 `pthread_once()` 调用 `initOnceFunc()`
```cpp
static void initOnceFunc() {
  // 读取 `~/.nccl.conf` 和 `/etc/nccl.conf` 中的环境变量并设置
  initEnv();
  // 加载 gdr 库
  initGdrCopy();
  // Always initialize bootstrap network
  NCCLCHECKGOTO(bootstrapNetInit(), initResult, exit);

  // NVTX: NVIDIA Tools Extension
  initNvtxRegisteredEnums();
exit:;
}
```

### `ncclCommInitRank[All/Config]()` 函数

调用 `ncclCommInitRankDev()` 函数实现。
```cpp
ncclResult_t ncclCommInitRank(ncclComm_t* newcomm, int nranks, ncclUniqueId commId, int myrank) {
  // Load the CUDA driver and dlsym hooks (can fail on old drivers)
  (void)ncclCudaLibraryInit();

  int cudaDev;
  ncclConfig_t config = NCCL_CONFIG_INITIALIZER;

  // 因此调用 ncclCommInitRank() 前需要 cudaSetDevice 指定 cuda device
  CUDACHECK(cudaGetDevice(&cudaDev));

  NvtxParamsCommInitRank payload{myrank, nranks, cudaDev};
  NVTX3_FUNC_WITH_PARAMS(CommInitRank, CommInitRankSchema, payload)

  NCCLCHECK(ncclCommInitRankDev(newcomm, nranks, commId, myrank, cudaDev, &config));
  return ncclSuccess;
}
```

`ncclComm_t` 是指向 `ncclComm` 的指针。
```cpp
typedef struct ncclComm* ncclComm_t;
```

`ncclAsyncJob` 是一个关键结构，后续的操作都会转为一个 job 链入全局链表 `ncclAsyncJobs` 中。
```cpp
struct ncclAsyncJob {
  struct ncclAsyncJob* next;
  pthread_t thread;
  ncclResult_t result;
  ncclResult_t(*func)(struct ncclAsyncJob*);
  void(*undo)(struct ncclAsyncJob*);
  void(*destructor)(void*);
  ncclGroupJobState_t state;
  uint32_t* abortFlag; /* point to comm abortFlag */
  uint32_t* abortFlagDev; /* point to comm abortFlagDev */
  uint32_t* childAbortFlag; /* point to child abortFlag */
  uint32_t* childAbortFlagDev; /* point to child abortFlagDev */
  ncclComm_t comm;
  int destroyFlag;
};
```

`ncclCommInitRankDev()` 为参数 `newcomm` 分配了空间，然后调用 `ncclAsyncLaunch()` 进行初始化。这里的 `job` 是 `ncclCommInitRankAsyncJob`，里面包含了 `ncclAsyncJob` 和额外的信息。之后也有其它类似的结构。
```cpp
// ncclCommInitRankFunc() 里给 Comm 初始化
NCCLCHECKGOTO(ncclAsyncLaunch(&job->base, ncclCommInitRankFunc, NULL, free, comm), res, fail);

ncclResult_t ncclAsyncLaunch(
  struct ncclAsyncJob* job,
  ncclResult_t(*func)(struct ncclAsyncJob*),
  void(*undo)(struct ncclAsyncJob*),
  void(*destructor)(void*), ncclComm_t comm
);

struct ncclCommInitRankAsyncJob {
  struct ncclAsyncJob base;
  ...
}
```

`ncclAsyncLaunch()` 会调用 `func(job)`，是否异步与 NCCL group 机制相关（异步即先链入全局链表 `ncclAsyncJobs`），详见之后内容。因此，这里的调用会执行 `ncclCommInitRankFunc()`，其中会初始化一些 `ncclComm` 结构中的字段，具体在 `commAlloc()` 函数之中。


### `ncclGroupStart()/ncclGroupEnd()` 函数
`ncclGroupStart()/ncclGroupEnd()` 可以将多个操作融合为一个。源码中是对 `ncclGroupStartInternal()/ncclGroupEndInternal()` 的简单封装。

`ncclGroupStartInternal()` 较为简单，将全局变量 `ncclGroupDepth` 递增。
```cpp
inline ncclResult_t ncclGroupStartInternal() {
  ncclGroupDepth++;
  return ncclSuccess;
}
```

`ncclGroupEndInternal()` 比较复杂，在 group 结束时执行 group 内的操作。具体流程见后文。

### Collective
定义在 `src/collectives.cc`，以 `ncclAllReduce()` 为例。
```cpp
NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct NvtxParamsAllReduce {
    size_t bytes;
    ncclRedOp_t op;
  };
  // Just pass the size of one message and not the total bytes sent/received.
  static constexpr nvtxPayloadSchemaEntry_t AllReduceSchema[] = {
    {0, NVTX_PAYLOAD_ENTRY_TYPE_SIZE, "Message size [bytes]"},
    {0, NVTX_PAYLOAD_ENTRY_NCCL_REDOP, "Reduction operation", nullptr, 0,
      offsetof(NvtxParamsAllReduce, op)}
  };
  NvtxParamsAllReduce payload{count * ncclTypeSize(datatype), op};
  NVTX3_FUNC_WITH_PARAMS(AllReduce, AllReduceSchema, payload)

  struct ncclInfo info = { ncclFuncAllReduce, "AllReduce",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS };

  // 将操作插入队列
  NCCLCHECK(ncclEnqueueCheck(&info));
  return ncclSuccess;
}
```

主要流程是配置好相关信息后，将操作通过 `ncclEnqueueCheck()` 入队。在 `ncclEnqueueCheck()` 主要通过 `taskAppend()` 将任务入队。
```cpp
NCCLCHECK(ncclGroupStartInternal());
...
NCCLCHECKGOTO(taskAppend(info->comm, info), ret, fail);
...
NCCLCHECK(ncclGroupEndInternal());
```

`taskAppend()` 主要任务是，将 info 信息转为 task，并链入 `ncclComm` 中的 `planner`。在此之前，将本 communicator 链入全局链表 `ncclGroupCommHead` 中。
```cpp
ncclGroupCommJoin(info->comm);
```

P2P 和 Collective 分为不同的队列。
```cpp
struct ncclKernelPlanner {
  struct Peer {
    bool sendSeen, recvSeen;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> sendQueue;
    struct ncclIntruQueue<struct ncclTaskP2p, &ncclTaskP2p::next> recvQueue;
  };
  struct ncclTaskCollSorter collSorter;
  ...
};
```

P2P 比较简单，直接链入对应队列，之后标记需要提前连接的 channel（将本 communicator 链入全局链表 `ncclGroupCommPreconnectHead`）。
```cpp
ncclIntruQueueEnqueue(isSendNotRecv ? &planner->peers[peer].sendQueue : &planner->peers[peer].recvQueue, p2p);
...
```

Collective 相对复杂，需要先把 task 放入 `collSorter`。（如果 `nRanks == 1`，说明只有一个 device，直接 launch 单个 device 的 cuda kernel即可）
```cpp
ncclTaskCollSorterInsert(&planner->collSorter, t, t->trafficBytes);
```

`ncclTaskCollSorter` 是个相对复杂的结构。它主体是一个 `ncclTaskColl` 链表，但根据 `trafficBytes` 分为了若干个 bin，每个 bin 都是上述链表的一个 slice，`bins` 数组保存了每个 bin 的头指针的地址。`ncclTaskCollSorterInsert()` 向其中插入一个 task，具体逻辑可以看 `src/include/comm.h`。
```cpp
struct ncclTaskCollSorter {
  ...
  // head -> tail 是一个完整的单链表，分为一段段的 bin，每个 bin 都是这个链表截取一段
  struct ncclTaskColl* head;
  struct ncclTaskColl* tail;
  ...
  // bins 储存了不同 bin 指向该 bin 头的地址，可能指向 head，也可能指向上个 bin 尾的 next
  struct ncclTaskColl** bins[BinCount];
};
```

在将 p2p/collective 插入 `planner` 之后，需要将 cuda stream 也链入 `planner`，这里要求所有 stream 的 graph 相同。
```cpp
struct ncclCudaStreamList {
  struct ncclCudaStreamList *next;
  cudaStream_t stream;
};

struct ncclKernelPlanner {
  ...
  // The list of user streams aggregated over all tasks present.
  struct ncclCudaStreamList* streams;
  // The most recent user stream. Ignored if streams==nullptr
  cudaStream_t streamRecent;
  ...
};
```

如此，`taskAppend()` 的调用就结束了，返回到 `ncclEnqueueCheck()`。而后者的函数体是一对 `ncclGroupStartInternal()/ncclGroupEndInternal()`。因此，在 `ncclGroupEndInternal()` 中，task 才会被真正提交到 device。

### `ncclGroupEndInternal()`
`ncclGroupEndInternal()` 的定义在 `src/group.cc` 中。

首先，将 `ncclGroupDepth` 减 1，如果结果不为 0，说明在一个嵌套 group 里，直接退出。
```cpp
if ((--ncclGroupDepth) > 0) goto exit;
```

如果为 0，就到了 group 结束阶段。判断这个 group 是否有任务。
```cpp
if (ncclGroupCommHead != nullptr || !ncclIntruQueueEmpty(&ncclAsyncJobs) || ncclGroupCommPreconnectHead != nullptr) {
    ...
}
```

如果有，构建一个 group job，即给全局变量 `ncclGroupJobMain` 赋值。`ncclGroupJob` 结构与前文 `ncclCommInitRankAsyncJob` 类似，都包含一个 `ncclAsyncJob` 和其它信息。
```cpp
__thread struct ncclGroupJob ncclGroupJobMain;
```

之后就是执行这个 job，逻辑在函数 `groupLaunch()` 中。NCCL 中有 blocking 设置，这里先以 blocking 为例。
```cpp
static ncclResult_t groupLaunch(struct ncclAsyncJob *job_, ncclSimInfo_t* simInfo = NULL);
static ncclResult_t groupLaunchNonBlocking(struct ncclAsyncJob *job_) {
  return groupLaunch(job_ /* estimatedTime = NULL */);
}
...
// in `ncclGroupEndInternal()`
/* blocking group */
NCCLCHECKGOTO(groupLaunch(&ncclGroupJobMainPtr->base, internalSimInfoPtr), ret, fail);
```

`groupLaunch()` 中，

1. 将之前 p2p 标记的 preconnect 生成 job 链入 `asyncJobsMain`（这里的 `asyncJobsMain` 就是全局的 `ncclAsyncJobs`）。
```cpp
if (!simInfo && groupCommPreconnectHeadMain != nullptr) {
    do {
        struct ncclPreconnectJob* job;
        NCCLCHECKGOTO(ncclCalloc(&job, 1), ret, fail);
        job->base.func = ncclP2PPreconnectFunc;
        job->base.undo = nullptr;
        job->base.destructor = free;
        job->base.state = ncclGroupJobRunning;
        job->base.abortFlag = comm->abortFlag;
        job->base.abortFlagDev = comm->abortFlagDev;
        job->comm = comm;
        ncclIntruQueueEnqueue(asyncJobsMain, &job->base);

        struct ncclComm* next = comm->preconnectNext;
        comm->preconnectNext = reinterpret_cast<struct ncclComm*>(0x1);
        comm = next;
    } while (comm != nullptr);
}
```

2. 调用 `asyncJobLaunch()` 完成 `asyncJobsMain`。 `asyncJobLaunch()` 的流程是给每一个 job 创建一个线程，异步执行，最后等待所有 job 执行完成。
```cpp
NCCLCHECKGOTO(asyncJobLaunch(asyncJobsMain, groupAbortFlag), ret, fail);
```

3. 调用 `ncclPrepareTasks()` 把 comm->planner->collSorter 的 Coll 都转为 task，之后完成 pre connect（支持 cumem 的请况下）。

4. 调用 `doLaunches(groupCommHeadMain)`，将 communicator 中的 task 在相应 cuda device 上启动。

`doLaunches()` 中，遍历 `groupCommHeadMain` 中的每个 communicator。对每个 comm，调用 `ncclLaunchPrepare(comm)` 把 p2p/coll task 转为 plan 并放到 planner->planQueue 里，之后遍历队列，依次 launch kernel`ncclLaunchKernel(comm, plan)`。在 launch 时，如果 `NCCL_LAUNCH_MODE` 为 `Group`，则需要在每个 launch 后增加一个 barrier `ncclCommIntraBarrierIn()`，如果为 `Parallel` 不用。

在 `ncclLaunchKernel()` 中会调用 `plan->kernelFn()`，这是一个函数指针，指向对应的 cuda kernel。这个指针在 `ncclPrepareTasks` 填写，
```cpp
// collective
plan->kernelFn = ncclDevKernelForFunc[task->devFuncId];
plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[task->devFuncId];

// p2p
plan->kernelFn = ncclDevKernelForFunc[ncclDevFuncId_P2p()];
plan->kernelSpecialized = ncclDevKernelForFuncIsSpecialized[ncclDevFuncId_P2p()];
```
`devFuncId` 来自于 `ncclDevFuncId()/ncclDevFuncId_P2p()`，在 `src/include/device.h` 中。
```cpp
extern int const ncclDevFuncIdCount;
extern int const ncclDevFuncRowToId[];
extern void* const ncclDevKernelForFunc[/*funcIndex*/];
extern bool const ncclDevKernelForFuncIsSpecialized[/*funcIndex*/];

inline int ncclDevFuncId(int coll, int devRedOp, int type, int algo, int proto) {
    ...
    return ncclDevFuncRowToId[row];
}

inline int ncclDevFuncId_P2p() { return ncclDevFuncRowToId[0]; }
```
这四个变量来自于 `src/device/generate.py` 生成的 `host_table.cc`，`ncclDevFuncId` 要与 `generate.py` 中的 `all_functions` 一致。除了 `host_table.cc`，`generate.py` 还会生成若干 `.cu` 文件，里面是 cuda kernel 的定义，这些定义来自于对 `src/device` 目录下头文件的引用，里面包含了一些模板。

至此，一个 group 执行完成。

#### blocking and non-blocking
全局变量 `ncclGroupBlocking`。
```cpp
__thread int ncclGroupBlocking = -1; /* default mode */
```
在 `ncclAsyncLaunch()` 和 `ncclGroupCommJoin()` 中会被赋值，要保证一个 group 的设置是一致的。

如果是 non-blocking，在 `ncclGroupEndInternal()` 中，`groupLaunch()` 会被提交到另一个线程执行，
```cpp
SYSCHECKGOTO(pthread_create(&ncclGroupJobMainPtr->base.thread, NULL, ncclAsyncJobMain, (void*)&ncclGroupJobMainPtr->base), ret, fail);

void* ncclAsyncJobMain(void* arg) {
  struct ncclAsyncJob* job = (struct ncclAsyncJob*)arg;
  job->result = job->func(job);
  if (job->result != ncclSuccess) {
    INFO(NCCL_INIT,"%s:%d -> %d [Async thread]", __FILE__, __LINE__, job->result);
  }
  __atomic_store_n(&job->state, ncclGroupJobDone, __ATOMIC_RELEASE);
  return arg;
}
```

工作线程通过 `ncclCommSetAsyncError()` 设置对应 communicator 的状态，主线程通过 `ncclCommGetAsyncError()` 获取。通过对 `comm->asyncResult` 的原子操作实现。
```cpp
ncclResult_t ncclCommSetAsyncError(ncclComm_t comm, ncclResult_t nextState) {
  if (nextState < 0 || nextState >= ncclNumResults || comm == NULL) {
    WARN("ncclCommSetAsyncError: error comm %p sets state %d", comm, nextState);
    return ncclInvalidArgument;
  }

  __atomic_store_n(&comm->asyncResult, nextState, __ATOMIC_RELEASE);
  return ncclSuccess;
}

ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t *asyncError) {
  NCCLCHECK(CommCheck(comm, "ncclGetAsyncError", "comm"));
  NCCLCHECK(PtrCheck(asyncError, "ncclGetAsyncError", "asyncError"));

  *asyncError = __atomic_load_n(&comm->asyncResult, __ATOMIC_ACQUIRE);
  if (*asyncError == ncclSuccess && comm->proxyState) *asyncError = __atomic_load_n(&comm->proxyState->asyncResult, __ATOMIC_ACQUIRE);
  return ncclSuccess;
}
```

例如，在工作线程挂起前，`ncclGroupEndInternal()` 调用 `ncclCommSetAsyncError()` 将状态设置为 `ncclInProgress`。
```cpp
...
ncclCommSetAsyncError(comm, ncclInProgress);
...
pthread_create(&ncclGroupJobMainPtr->base.thread, NULL, ncclAsyncJobMain, (void*)&ncclGroupJobMainPtr->base);
...
```

### bootstrap 网络
bootstrap 网络顾名思义，是为了交换一些基本信息比如地址什么的。

bootstrap 网络的初始化通过 `bootstrapNetInit()`，这个函数只在 `ncclInit()` 调用。按 [nccl 官网的例子](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/examples.html#example-2-one-device-per-process-or-thread)，rank 为 0 的会调用 `ncclGetUniqueId()` 从而调用 `ncclInit()`，否则走 `ncclCommInitRank() -> ncclCommInitRankDev() -> ncclInit()`。`bootstrapNetInit()` 的作用是获得 bootstrap 网络的地址端口信息，并保存在全局变量 `bootstrapNetIfAddr` 和 `bootstrapNetIfName` 中。`ncclUniqueId` 是一个 128 字节的空间，语义上就是一个 `ncclBootstrapHandle`，里面保存地址信息。调用 `ncclGetUniqueId()` 的 root 就通过 MPI 将 root 的 bootstrap 地址传递给其它 node。

```cpp
struct ncclBootstrapHandle {
  uint64_t magic;
  union ncclSocketAddress addr;
};
```

在`bootstrapGetUniqueId()/ncclCommInitRankDev()` 中，对于 `rank=0` 的节点要建立 bootstrap 根，即 `bootstrapCreateRoot()` 函数。

```cpp
ncclResult_t bootstrapCreateRoot(struct ncclBootstrapHandle* handle, bool idFromEnv) {
  struct ncclSocket* listenSock;
  struct bootstrapRootArgs* args;
  pthread_t thread;

  NCCLCHECK(ncclCalloc(&listenSock, 1));
  NCCLCHECK(ncclSocketInit(listenSock, &handle->addr, handle->magic, ncclSocketTypeBootstrap, NULL, 0));
  NCCLCHECK(ncclSocketListen(listenSock));
  NCCLCHECK(ncclSocketGetAddr(listenSock, &handle->addr));

  NCCLCHECK(ncclCalloc(&args, 1));
  args->listenSock = listenSock;
  args->magic = handle->magic;
  NEQCHECK(pthread_create(&thread, NULL, bootstrapRoot, (void*)args), 0);
  ncclSetThreadName(thread, "NCCL BootstrapR");
  NEQCHECK(pthread_detach(thread), 0); // will not be pthread_join()'d
  return ncclSuccess;
}
```

这个函数启动了一个新线程 `bootstrapRoot()`，在后台等待与其它 node 通信。首先等待其它 node 向它发送各自的地址，保存在 `rankAddresses/rankAddressesRoot` 中。当接收所有 node 之后，将 ring AllGather 中的下一个地址发送给各个 node。（`rankAddresses/rankAddressesRoot` 对应两个 socket，一个用于和 root 通信，一个用于 ring）

在 `ncclCommInitRankDev() -> ncclCommInitRankFunc()` 中，在 `commAlloc()` 之后会执行 `bootstrapInit()` 函数。这个函数用于构建 bootstrap 网络。有若干步骤：

1. 向 root 发送自身地址；
2. 接收 root 发送的 ring 中 next node 的地址；
3. 通过 AllGather 获得所有地址，放入 `peerCommAddresses` ；
4. 建立服务代理等等（这些还不清楚用来干什么）

bootstrap 网络句柄保存在 `ncclComm->bootstrap` 中。

### Main 网络
