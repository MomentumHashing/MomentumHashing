#include <iostream>
#include <memory.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <functional>
#include <ctime>
#include <map>
#include <set>
#include "ARROWHASH.cuh"
#include <algorithm>
#include <string>
#include <time.h>
#include <thread>
#include <unistd.h>
#include <sys/time.h>
#include <mutex>
#include "gputimer.h"
#include <omp.h>

using namespace std;
__global__ void printresult(int*result){
    for(int i=0;i<5;++i)
      printf("%d\n",result[i]);

}

int main(int argc, char **argv)
{
  set<Key>all_id;
    generate_seeds();
    cudaSetDeviceFlags(cudaDeviceMapHost);
  Key* KeyArray;
  cudaMallocHost (&KeyArray, totalnum*sizeof(Key));
  
  //cudaHostGetDevicePointer (&GPUKeyArray, KeyArray, 0);

  ArrowHash*table=new ArrowHash;
  FpTable*fptable= new FpTable;
  FpTable*GPUfptable;


  int id=0;
  while(all_id.size() < totalnum){
    Key a(to_string(id));
      all_id.insert(a);
      ++id;

  }
  



  int ii=0;
  for(set<Key>::iterator iter = all_id.begin(); iter != all_id.end(); iter++){
    KeyArray[ii]=(*iter);
    ++ii;
  }
  Key*TempKeyArray;
  TempKeyArray=new Key[all_id.size()];
  memcpy(TempKeyArray,KeyArray,totalnum*sizeof(Key));

  for(int ompnum=0;ompnum<=6;++ompnum){
     omp_set_num_threads((1<<ompnum));
     table->Clear();
     fptable->Clear();
double start=omp_get_wtime( );
#pragma omp parallel for schedule(dynamic,1024)
      for(int i=0;i<totalnum;++i){

        insert(fptable,table,KeyArray[i]);
    }


    double end=omp_get_wtime( );
  cout<<"CPU threads = "<<(1<<ompnum)<<endl;
  cout<<"inserted mips ="<<double(totalnum)/(1000000 * float(end - start))<<endl;
  cout<<"==================================="<<endl;
  }
  
 
  
  
  cudaMalloc(&GPUfptable, sizeof(FpTable));
  cudaMemcpy(GPUfptable, fptable, sizeof(FpTable), cudaMemcpyHostToDevice);
  delete fptable;
  
    for(int ompnum=0;ompnum<=6;++ompnum){
    int*cpuresults;
    cudaMallocHost (&cpuresults, (totalnum+totalnum/8)*sizeof(int));
    int*cpupos=cpuresults+totalnum;
    Key* GPUKeyArray;
    int*results;
    //cudaHostGetDevicePointer (&results, cpuresults, 0);
   // int*pos=;

    cudaMalloc(&results, (totalnum+totalnum/8)*sizeof(int));//one int can represent the hash seed of 16 keys
    //cudaMalloc(&pos, totalnum*sizeof(int)/8);
    cudaMalloc(&GPUKeyArray, totalnum*sizeof(Key));
    int querynum=0;
    omp_set_num_threads((1<<ompnum));
    memcpy(KeyArray,TempKeyArray,totalnum*sizeof(Key));
    cudaStream_t stream[BatchNum];
    for(int i=0;i<BatchNum;++i)
      cudaStreamCreate(&stream[i]);
    
    double starttime=omp_get_wtime( );
    double gpustarttime=omp_get_wtime( );
    //double gpucopystarttime1=omp_get_wtime( );
    for(int streamID=0;streamID<BatchNum;++streamID){
    
    cudaMemcpyAsync(GPUKeyArray+streamID*BatchSize, KeyArray+streamID*BatchSize, BatchSize*sizeof(Key), cudaMemcpyHostToDevice,stream[streamID]);
    
    }
    for(int streamID=0;streamID<BatchNum;++streamID){
    
    
    GPUquery<<<blocknum,threadnum,0,stream[streamID]>>>(GPUfptable,GPUKeyArray,GPUseeds,results,streamID);
    }
    for(int streamID=0;streamID<BatchNum;++streamID){
    
    cudaMemcpyAsync(cpuresults+streamID*(BatchSize), results+streamID*(BatchSize), (BatchSize)*sizeof(int), cudaMemcpyDeviceToHost,stream[streamID]);
    cudaMemcpyAsync(cpupos+streamID*(BatchSize/8), results+totalnum+streamID*(BatchSize/8), (BatchSize/8)*sizeof(int), cudaMemcpyDeviceToHost,stream[streamID]);

    }

   



    double gpuendtime=omp_get_wtime( );

    double cpustarttime=omp_get_wtime( );


  for(int streamID=0;streamID<BatchNum;++streamID){
  cudaStreamSynchronize(stream[streamID]);
    #pragma omp parallel for schedule(dynamic,1024)
      for(int i=streamID*BatchSize;i<streamID*BatchSize+BatchSize;++i){

     int temp=((cpupos[(i>>3)]>>((i%8)<<2))&(15));
     int find=0;
     
      if(cpuresults[i]<0){

        for(int j=temp;j<SlotPerBucket;++j){
          if(KeyArray[i]==table->Toplevel[-cpuresults[i]].key[j]){
            KeyArray[i].clear(); 
           find++; 
            break;;
          }
        } 
       
      }
     if(cpuresults[i]>0||!find){
       
      for(int j=temp;j<SlotPerBucket;++j){
          if(KeyArray[i]==table->Bottomlevel[cpuresults[i]].key[j]){
           KeyArray[i].clear();
           // ++querynum;
           break;
          }    
        }
      }
    
    }


    }











    double cpuendtime=omp_get_wtime( );
    double endtime=omp_get_wtime( );
    double diff = endtime-starttime;
    
    printf("CPU Time elapsed = %g s\n", cpuendtime-cpustarttime); // 输出
    printf("GPU Calculate Time elapsed = %g s\n", gpuendtime-gpustarttime); // 输出
    printf("Time elapsed = %g s\n", diff); // 输出
  cout<<"query mips="<<double(all_id.size())/(1000000 * (diff))<<endl;
  cout<<querynum<<endl;
  cout<<"CPU threads = "<<(1<<ompnum)<<endl;
  cout<<"<<<Block, Thread>>> = "<<"<<<"<<blocknum<<" "<<threadnum<<">>>"<<endl;
  cout<<"==================================="<<endl;
  cudaFree(results);
  cudaFree(GPUKeyArray);
  cudaFreeHost(cpuresults);
  for(int i=0;i<BatchNum;++i)
      cudaStreamDestroy(stream[i]);
  cout<<"finish"<<endl;

    }
   

  return 0;
}
