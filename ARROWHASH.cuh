#ifndef _CMSKETCH_H_VERSION_2
#define _CMSKETCH_H_VERSION_2
#include <sstream>
#include <cstring>
#include <algorithm>
#include <cstdint>
#include <iostream>
#include <memory.h>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <functional>
#include <ctime>
#include <map>
#include <set>
#include <bitset>
#include <mutex>
#include <cuda.h>
#define MAXINT 9223372036854775807
#define stash_size 4
#define SlotPerBucket 12
#define w 12
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define fenzi 3
#define fenmu 4
#define blocknum 128
#define threadnum 128
#define calshortprint(fingerprint1,fingerprint2, offsett) (fingerprint1^(fingerprint2 << (64-offsett)|(fingerprint2>>offsett))) & ((1 << w) - 1)

using namespace std;


#define TotalBucketinSmallLevel (1<<17)
#define TotalBucketinBigLevel (1<<18)
#define insertnum 3*TotalBucketinSmallLevel*SlotPerBucket
#define BatchNum 16
#define ratio 4
#define totalnum ratio*insertnum
#define BatchSize totalnum/BatchNum

uint32_t nooffset=0;
uint32_t failnum=0;
int seed[5]={0};

int*GPUseeds;
template <size_t N>
struct FixedLengthStr
{
    static const string empty_s;
    char x[N];
    static const size_t size = N;

    
	CUDA_CALLABLE_MEMBER FixedLengthStr() : x{'\0'} {}
    
	CUDA_CALLABLE_MEMBER FixedLengthStr(const char* s) { memcpy(x, s, N); }
    
	FixedLengthStr(const string &s) { memcpy(x, s.c_str(), N); }
    
	CUDA_CALLABLE_MEMBER FixedLengthStr(const FixedLengthStr &other) { memcpy(x, other.x, N); }
    
	CUDA_CALLABLE_MEMBER FixedLengthStr& operator=(const FixedLengthStr &other) {
        memcpy(x, other.x, N);
        return *this;
    }
    
	CUDA_CALLABLE_MEMBER bool operator!=(const FixedLengthStr &other) const { return memcmp(x, other.x, N); }
    
	 bool operator==(const FixedLengthStr &other) const { return !(*this!=other); }
    
	CUDA_CALLABLE_MEMBER bool operator==(const char* s) const { return !memcmp(x, s, N); }
    CUDA_CALLABLE_MEMBER bool empty() const { return (*this==empty_s.c_str()); }
    CUDA_CALLABLE_MEMBER void clear() { memset(x, 0, N); }

};



void generate_seeds()
{
    seed[0]=11;
    seed[1]=17;
    seed[2]=111113;
    seed[3]=19;
    seed[4]=23;
    cudaMalloc(&GPUseeds, 5*sizeof(int));
    cudaMemcpy(GPUseeds, seed, 5*sizeof(int), cudaMemcpyHostToDevice);
}


template <size_t N> const string FixedLengthStr<N>::empty_s(N, '\0');
typedef FixedLengthStr<16> Key;
typedef FixedLengthStr<16> Data;

CUDA_CALLABLE_MEMBER bool operator<(Key an, Key bn)
{
	for (int i = 0; i < 16; i++)
	{
		if (bn.x[i] < an.x[i])
		{
			return true;
		}
		else if (bn.x[i] > an.x[i])
		{
			return false;
		}
	}
	return false;
}


CUDA_CALLABLE_MEMBER unsigned long long MurmurHash64B ( const void * key, int len, unsigned int seed )
{
	const unsigned int m = 0x5bd1e995;
	const int r = 24;
 
	unsigned int h1 = seed ^ len;
	unsigned int h2 = 0;
 
	const unsigned int * data = (const unsigned int *)key;
 
	while(len >= 8)
	{
		unsigned int k1 = *data++;
		k1 *= m; k1 ^= k1 >> r; k1 *= m;
		h1 *= m; h1 ^= k1;
		len -= 4;
 
		unsigned int k2 = *data++;
		k2 *= m; k2 ^= k2 >> r; k2 *= m;
		h2 *= m; h2 ^= k2;
		len -= 4;
	}
 
	if(len >= 4)
	{
		unsigned int k1 = *data++;
		k1 *= m; k1 ^= k1 >> r; k1 *= m;
		h1 *= m; h1 ^= k1;
		len -= 4;
	}
 
	switch(len)
	{
	case 3: h2 ^= ((unsigned char*)data)[2] << 16;
	case 2: h2 ^= ((unsigned char*)data)[1] << 8;
	case 1: h2 ^= ((unsigned char*)data)[0];
			h2 *= m;
	};
	h1 ^= h2 >> 18; h1 *= m;
	h2 ^= h1 >> 22; h2 *= m;
	h1 ^= h2 >> 17; h1 *= m;
	h2 ^= h1 >> 19; h2 *= m;
	unsigned long long h = h1;
	h = (h << 32) | h2;
	return h;
} 


CUDA_CALLABLE_MEMBER uint64_t defaultHash(const Key &k, int seed) {
        uint64_t h;
        h=MurmurHash64B(k.x, k.size, seed);
        return h;
 }







struct Bucket{
	Key key[SlotPerBucket];
	Data val[SlotPerBucket];
	uint64_t Avaoffset;//availavle offset
	uint32_t num=0;
	CUDA_CALLABLE_MEMBER Bucket(){
		
			Avaoffset=MAXINT;
		

	}
	void Clear(){
		Avaoffset=MAXINT;
		memset(key,0,sizeof(key));
		num=0;
	}
	
};

struct FPBucket{
	uint32_t offset=0;
	uint32_t FP[SlotPerBucket]={0};
	void Clear(){
		offset=0;
		memset(FP,0,sizeof(FP));
	}
	

};



/*
bool collision_elimination(Bucket*maintable, FPBucket*table,uint32_t fingerprint, uint32_t index){
     for(int shift=0;shift<SlotPerBucket;++shift){
		int i;
	    uint64_t d=1;
        uint32_t shortprint[2] = {0};
		uint64_t longprint[2] = {0};
		for (int j = 0; j < d; j++)
			longprint[j] = defaultHash(maintable[index].key[shift],seed[4]);
		longprint[d] = fingerprint;
		for (i = 0; i < 64; i++)
		{
			if (!(maintable[index].Avaoffset[shift] & (1 << i)))
				continue;
			for (int j = 0; j <= d; j++)
				shortprint[j] = int(calshortprint(longprint[j], i));
			for (int j = 0; j < d; j++)
				if (shortprint[j] == shortprint[d])
				{
					maintable[index].Avaoffset[shift] &= ~(1 << i);
					break;
				}
		}
		if (!(maintable[index].Avaoffset[shift] & (1 << table[index].offset[shift])))
		{
			for (int j = table[index].offset[shift] + 1; j < 64; j++)
				if (maintable[index].Avaoffset[shift] & (1 << j))
				{
					table[index].offset[shift] = j;
					break;
				}
			if (maintable[index].Avaoffset[shift] & (1 << table[index].offset[shift]))
			{
				for (int j = 0; j < d; j++)
					table[index].FP[shift] = uint32_t(calshortprint(longprint[j], table[index].offset[shift]));
			}
			else
			{
		
				return false;
			}
		}

     }
    return true;

	    
}*/




bool collision_elimination(Bucket*maintable, FPBucket*table,uint32_t fingerprint1,uint32_t fingerprint2, uint32_t index){
     
		int i;
	    uint64_t d=SlotPerBucket;
        uint32_t shortprint[13] = {0};
		uint64_t longprint1[13] = {0};
		uint64_t longprint2[13] = {0};
		for (int j = 0; j < d; j++){
			longprint1[j] = defaultHash(maintable[index].key[j],seed[3]);
			longprint2[j] = defaultHash(maintable[index].key[j],seed[0]);
		}
		longprint1[d] = fingerprint1;
		longprint2[d] = fingerprint2;
		for (i = 0; i < 64; ++i)
		{
			if (!(maintable[index].Avaoffset & (1 << i)))
				continue;
			for (int j = 0; j <= d; ++j)
				shortprint[j] = int(calshortprint(longprint1[j],longprint2[j], i));
			for (int j = 0; j < d; ++j)
				if (shortprint[j] == shortprint[d])
				{
					maintable[index].Avaoffset &= ~(1 << i);
					break;
				}
		}
		if (!(maintable[index].Avaoffset & (1 << table[index].offset)))
		{
			for (int j = table[index].offset + 1; j < 64; ++j)
				if (maintable[index].Avaoffset & (1 << j))
				{
					table[index].offset = j;
					break;
				}
			if (maintable[index].Avaoffset & (1 << table[index].offset))
			{
				for (int j = 0; j < d; j++)
					table[index].FP[j] = uint32_t(calshortprint(longprint1[j],longprint2[j], table[index].offset));
			}
			else
			{
		
				return false;
			}
		}
		

    return true;

	    
}









class FpTable{
public:
	FPBucket Toplevel[TotalBucketinSmallLevel];
	FPBucket Bottomlevel[TotalBucketinBigLevel];
	void Clear(){
		for(int i=0;i<TotalBucketinSmallLevel;++i)
			Toplevel[i].Clear();
		for(int i=0;i<TotalBucketinBigLevel;++i)
			Bottomlevel[i].Clear();



	}

};
class ArrowHash{
public:
	Bucket Toplevel[TotalBucketinSmallLevel];
	Bucket Bottomlevel[TotalBucketinBigLevel];
	void Clear(){
		for(int i=0;i<TotalBucketinSmallLevel;++i)
			Toplevel[i].Clear();
		for(int i=0;i<TotalBucketinBigLevel;++i)
			Bottomlevel[i].Clear();



	}


};


/*

bool insert(FpTable*a,ArrowHash*b,const Key &k){
	//insert top level
	uint32_t index=defaultHash(k,seed[0]) % (fenzi*TotalBucketinSmallLevel/fenmu);
	uint64_t fingerprint=defaultHash(k,seed[4]);
	if(b->Toplevel[index].num<SlotPerBucket){
		a->Toplevel[index].FP[b->Toplevel[index].num]=uint32_t(calshortprint(uint64_t(fingerprint), 0));
		b->Toplevel[index].key[b->Toplevel[index].num]=k;
		++b->Toplevel[index].num;

		
	
	
		return true;
	}
	
	else{
	    
        if(!collision_elimination(b->Toplevel,a->Toplevel,fingerprint,index)){
			//cout<<fingerprint<<"    "<<defaultHash(b->Toplevel[index].key[i],seed[1])<<endl;
			nooffset++;
			return false;
		}

	    
		
		//index=a->Toplevel[index].nextpos;
		index=defaultHash(k,seed[1])%((fenmu-fenzi)*TotalBucketinSmallLevel/fenmu)+(fenzi*TotalBucketinSmallLevel/fenmu);
		if(b->Toplevel[index].num<SlotPerBucket){
		a->Toplevel[index].FP[b->Toplevel[index].num]=uint32_t(calshortprint(uint64_t(fingerprint), 0));
		b->Toplevel[index].key[b->Toplevel[index].num]=k;
		++b->Toplevel[index].num;
	
	
		return true;

	}
	}


	
	if(!collision_elimination(b->Toplevel,a->Toplevel,fingerprint,index)){
		
		nooffset++;
		return false;

	}


    index=defaultHash(k,seed[2])%(fenzi*TotalBucketinBigLevel/fenmu);
    
	if(b->Bottomlevel[index].num<SlotPerBucket){
		a->Bottomlevel[index].FP[b->Bottomlevel[index].num]=uint32_t(calshortprint(fingerprint, 0));
		b->Bottomlevel[index].key[b->Bottomlevel[index].num]=k;
		++b->Bottomlevel[index].num;
	
		
		return true;
	}
	else{

		
		if(!collision_elimination(b->Bottomlevel,a->Bottomlevel,fingerprint,index)){
			//cout<<fingerprint<<"    "<<defaultHash(b->Bottomlevel[index].key[i],seed[1])<<endl;
			nooffset++;
			return false;
			//cout<<"3"<<endl;
			//return false;
		}
	
		index=defaultHash(k,seed[3])%((fenmu-fenzi)*TotalBucketinBigLevel/fenmu)+(fenzi*TotalBucketinBigLevel/fenmu);
		//cout<<index<<"     "<<a->Bottomlevel[index].num<<endl;
		if(b->Bottomlevel[index].num<SlotPerBucket){
		a->Bottomlevel[index].FP[b->Bottomlevel[index].num]=uint32_t(calshortprint(fingerprint, 0));
		b->Bottomlevel[index].key[b->Bottomlevel[index].num]=k;
		++b->Bottomlevel[index].num;
		
	
		return true;

	}
	}


    return false;




	}


*/



mutex LockA[TotalBucketinSmallLevel];
mutex LockB[TotalBucketinBigLevel];


void insert(FpTable*a,ArrowHash*b,const Key &k){
	//insert top level
	uint32_t index=defaultHash(k,seed[1]) % TotalBucketinSmallLevel;
	uint64_t fingerprint2=defaultHash(k,seed[0]);
	uint64_t fingerprint1=defaultHash(k,seed[3]);
	LockA[index].lock();

	if(b->Toplevel[index].num<SlotPerBucket){
		a->Toplevel[index].FP[b->Toplevel[index].num]=uint32_t(calshortprint(fingerprint1,fingerprint2, a->Toplevel[index].offset));
		b->Toplevel[index].key[b->Toplevel[index].num]=k;
		++b->Toplevel[index].num;
		LockA[index].unlock();
		return;
	}
	uint32_t index2=(defaultHash(k,seed[2])%TotalBucketinBigLevel);
	LockB[index2].lock();
	if(b->Bottomlevel[index2].num<SlotPerBucket)
	{

		if(!collision_elimination(b->Toplevel,a->Toplevel,fingerprint1,fingerprint2,index)){
			//cout<<fingerprint<<"    "<<defaultHash(b->Toplevel[index].key[i],seed[1])<<endl;
			nooffset++;
			LockA[index].unlock();
			LockB[index2].unlock();
			return;
		}
		LockA[index].unlock();
		
		//cout<<index<<"     "<<a->Bottomlevel[index].num<<endl;
		a->Bottomlevel[index2].FP[b->Bottomlevel[index2].num]=uint32_t(calshortprint(fingerprint1,fingerprint2, a->Bottomlevel[index2].offset));
		b->Bottomlevel[index2].key[b->Bottomlevel[index2].num]=k;
		++b->Bottomlevel[index2].num;
		
		LockB[index2].unlock();
		return;

	}
	LockB[index2].unlock();

		uint32_t MinLoad=SlotPerBucket;
		int MinPos=0;
		int MinIndex=-1;
		Key TempKey;
		uint64_t Tempfingerprint1;
		uint64_t Tempfingerprint2;
		for(int i=0;i<SlotPerBucket;++i){
			int Tempindex=defaultHash(b->Toplevel[index].key[i],seed[2])%TotalBucketinBigLevel;
			if(Tempindex!=MinIndex)
				LockB[Tempindex].lock();
			else
				continue;
			if(b->Bottomlevel[Tempindex].num<MinLoad){
				MinLoad=min(MinLoad,b->Bottomlevel[Tempindex].num);
				MinPos=i;
				if(MinIndex!=-1)
					LockB[MinIndex].unlock();
				MinIndex=Tempindex;

				if(MinLoad==0)
					break;
				continue;
			}
			LockB[Tempindex].unlock();
			}
			//find
		if(MinLoad<SlotPerBucket){

		TempKey=b->Toplevel[index].key[MinPos];
		Tempfingerprint2=defaultHash(TempKey,seed[0]);
		Tempfingerprint1=defaultHash(TempKey,seed[3]);
		a->Toplevel[index].FP[MinPos]=uint32_t(calshortprint(fingerprint1,fingerprint2, a->Toplevel[index].offset));
		b->Toplevel[index].key[MinPos]=k;
		
		if(!collision_elimination(b->Toplevel,a->Toplevel,Tempfingerprint1,Tempfingerprint2,index)){
			nooffset++;
			LockB[MinIndex].unlock();
			LockA[index].unlock();
			return;
		}
		LockA[index].unlock();

		int Tempindex=defaultHash(TempKey,seed[2])%TotalBucketinBigLevel;
		a->Bottomlevel[Tempindex].FP[b->Bottomlevel[Tempindex].num]=uint32_t(calshortprint(Tempfingerprint1,Tempfingerprint2, a->Bottomlevel[Tempindex].offset));
		b->Bottomlevel[Tempindex].key[b->Bottomlevel[Tempindex].num]=TempKey;
		++b->Bottomlevel[Tempindex].num;
		LockB[MinIndex].unlock();
		return;
		}	

	
	LockA[index].unlock();	
    return;

	}







__global__ void GPUquery(FpTable*a, Key*karray,int*GPUseeds,int*results,int streamID,int myinterval=BatchSize/(blocknum*threadnum)){
	int*results_=results+streamID*(BatchSize);
	Key*karray_=karray+streamID*BatchSize;
    int indexx=blockDim.x*blockIdx.x+threadIdx.x;
    int*pos=results+totalnum+streamID*BatchSize/8;
    //
    for(int jj=indexx*myinterval;jj<indexx*myinterval+myinterval;++jj){
    Key k=karray_[jj];
    int flag=0;
    uint32_t index=defaultHash(k,GPUseeds[1]) % TotalBucketinSmallLevel;
	uint64_t fingerprint2=defaultHash(k,GPUseeds[0]);
	uint64_t fingerprint1=defaultHash(k,GPUseeds[3]);
    for(int i=0;i<SlotPerBucket;++i){
		if(a->Toplevel[index].FP[i]==uint32_t(calshortprint(fingerprint1,fingerprint2, a->Toplevel[index].offset))){
			results_[jj]=-index;
			pos[(jj>>3)]|=(i<<(4*(jj%8)));
			++flag;
			break;
		}
	}
	if(flag)
		continue;
    index=defaultHash(k,GPUseeds[2])%TotalBucketinBigLevel;
    for(int i=0;i<SlotPerBucket;++i){
		if(a->Bottomlevel[index].FP[i]==uint32_t(calshortprint(fingerprint1,fingerprint2, a->Bottomlevel[index].offset))){
			results_[jj]=index;
			pos[(jj>>3)]|=(i<<(4*(jj%8)));
			
			break;
		}
	}
    }
	}


#endif