/*
 * main.cc
 *
 *  Created on: Apr 9, 2014
 *      Author: rajmohan
 */

#include <stdio.h>
#include <iostream>
#include <mpi.h>
#include <math.h>
#include <vector>
#include <cstring>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <utility>
#include "util.h"
#include "io.h"

#define MAX_EPOCH 20

using std::string;
using namespace std;

int num_total_,num_pos_,num_neg_;
int num_test_total_,num_test_pos_,num_test_neg_;


// Structure storing properties of a feature, including id and weight.
struct Feature {
  int id;
  double weight;
};

// structure storing the properties of a document sample, including its document id,
// class label, the square of its two norm and all its features.
struct Sample {
  int id;
  int label;
  double two_norm_sq;
  vector<Feature> features;
};

struct AVector {
vector<Feature> features;
};

vector<Sample> train_samples_;
vector<Sample> test_samples_;
double training_time;

AVector weight_vector;

void train_perceptron(void);
bool training_method(void);
bool read_test_data(void);
void test_perceptron(void);
void train_mira(void);
void train_opa(void);
size_t GetPackSize(const AVector & sample);
size_t PackSample(char *&buffer, const AVector &sample);
size_t UnpackSample(AVector *&sample, const char *buffer);
double GetInnerProduct(const Sample& a, AVector w);
Sample* ScalarMultiplyVector(const Sample& a,double scalar);
AVector* ScalarMultiplyAVector(AVector a,double scalar);
AVector* AddTwoVectors(Sample& a, AVector w);
AVector* AddTwoAVectors(AVector a, AVector w);
size_t GetPackSize(const AVector & sample);
size_t PackSample(char *&buffer, const AVector &sample);
size_t UnpackSample(AVector *&sample, const char *buffer);
const Sample* GetLocalSample(int local_row_index);
const Sample* GetLocalTestSample(int local_row_index);
void recv_weight_vector_from_all(void);
void AddToW(const Sample &sample);
void send_weight_vector_to_all(void);
int myid, num_processors;

int main(void)
{
	int myid;

	training_time = 0.0;

	MPI_Init(NULL, NULL);
	MPI_Comm_size(MPI_COMM_WORLD, &num_processors); // no of processors
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	
	read_test_data();

	MPI_Barrier(MPI_COMM_WORLD);
	training_time -= MPI_Wtime();

	training_method();

	MPI_Barrier(MPI_COMM_WORLD);
	training_time += MPI_Wtime();
	
	test_perceptron();
  MPI_Finalize();
  return  0;
}

bool training_method(void)
{
	
	string train_file_path = "/home/rajmohan.c/project/perceptron/datasets/covtype/covtype.libsvm.binary.scale/covtype.libsvm_train.binary.scale";

	const char *filename = train_file_path.c_str();

	File* file = File::Open(filename, "r");
	if (file == NULL)
	{
	    cerr << "Cannot find file " << filename << endl;
	    MPI_Finalize();
	    return 0;
	}

   	string line;
	  int num_local_pos = 0;
	  int num_local_neg = 0;

	  while (file->ReadLine(&line))
	  {
	    // If the sample should be assigned to this processor
	    if (num_total_ % num_processors == myid)
	    {
	      int label = 0;
	      const char* start = line.c_str();
	      // Extracts the sample's class label
	      if (!SplitOneIntToken(&start, " ", &label))
	      {
	        cerr << "Error parsing line: " << num_total_ + 1 << endl;
	        return false;
	      }

	      // Gets the local number of positive and negative samples
	      if (label == 1)
	      {
	        ++num_local_neg;
	      }
	      else if (label == 2)
	      {
	        ++num_local_pos;
	      } else
	      {
	        cerr << "Unknow label in line: " << num_total_ + 1 << label;
	        return false;
	      }

	      // Creates a "Sample" and add to the end of samples_
	      train_samples_.push_back(Sample());
	      Sample& sample = train_samples_.back();
	      sample.label = label;
	      sample.id = num_total_;  // Currently num_total_ == sample id
	      sample.two_norm_sq = 0.0;

	      // Extracts the sample's features
	      vector<pair<string, string> > kv_pairs;
	      SplitStringIntoKeyValuePairs(string(start), ":", " ", &kv_pairs);
	      vector<pair<string, string> >::const_iterator pair_iter;
	      for (pair_iter = kv_pairs.begin(); pair_iter != kv_pairs.end();++pair_iter)
	      {
	        Feature feature;
	        feature.id = atoi(pair_iter->first.c_str());
	        feature.weight = atof(pair_iter->second.c_str());
	        sample.features.push_back(feature);
	        sample.two_norm_sq += (feature.weight * feature.weight);
	      }
	    }
	    ++num_total_;
	  }
	  file->Close();
	  delete file;

	  // Get the global number of positive and negative samples
	  int local[2];
	  int global[2];
	  local[0] = num_local_pos;
	  local[1] = num_local_neg;
	  memset(global, 0, sizeof(global[0] * 2));
	  MPI_Barrier(MPI_COMM_WORLD);
	  MPI_Allreduce(local, global, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	  num_pos_ = global[0];
	  num_neg_ = global[1];

/*	  if (myid == 0)
	  {
	      cout << "Total: " << num_total_
	                << "  Positive: " << num_pos_
	                << "  Negative: " << num_neg_ << endl;
	  }
*/
		// initialize the weight vector
		for(int i = 1 ; i <= 54 ; i++)
		{
			Feature feature;
			feature.id = i;
			feature.weight = 0;
			//		feature.weight = (double)rand()/(double)RAND_MAX; // for MIRA
			weight_vector.features.push_back(feature);
		}

		// enable algorithm accordingly
	 	train_perceptron();
//		train_mira(); 		//random init w. dont forget
//		train_opa();

}

// sends weight vector to all nodes from master after finding average
void send_weight_vector_to_all(void)
{
	int num_processors, myid;
	char *buffer = NULL;
	int buff_size,np;

	MPI_Comm_size(MPI_COMM_WORLD, &num_processors); // no of processors
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); // current processor id
	
	MPI_Barrier(MPI_COMM_WORLD);

	if(myid == 0)
	{
		// broadcast Avg_W to all nodes			
		buff_size = PackSample(buffer,weight_vector);
		for(np = 1; np < num_processors;np++)
		{
			MPI_Send(buffer, buff_size, MPI_BYTE, np,0, MPI_COMM_WORLD);
		}

	}
	else
	{
		// Get Avg_W and update locally		
		int rcv_buff_size = 656;
		char *rcv_buffer = NULL;		
		AVector *anewvector;
		anewvector = NULL;

		rcv_buffer = new char[rcv_buff_size];
		MPI_Recv(rcv_buffer, rcv_buff_size, MPI_BYTE,0,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);

		UnpackSample(anewvector, rcv_buffer);

		vector<Feature>::const_iterator it1 = anewvector->features.begin();
		while(it1 != anewvector->features.end())
		{
			weight_vector.features[(it1->id)-1].id = it1->id;
			weight_vector.features[(it1->id)-1].weight = it1->weight;
			it1++;
		}
	}
}
// Processor 0 gets weight vector from all processors and finds average weight vector
void recv_weight_vector_from_all(void)
{
	char *buffer = NULL;
	int buff_size,np;
	int num_processors, myid;
	double reg_prmtr;

	MPI_Comm_size(MPI_COMM_WORLD, &num_processors); // no of processors
	MPI_Comm_rank(MPI_COMM_WORLD, &myid); // current processor id
	reg_prmtr = (1.0/num_processors);

	buff_size = PackSample(buffer,weight_vector);

	MPI_Barrier(MPI_COMM_WORLD);

	if(myid != 0)
	{
		MPI_Send(buffer, buff_size, MPI_BYTE, 0,0, MPI_COMM_WORLD);
	}
	else
	{
		int rcv_buff_size = 656;	// 54 dim
		char *rcv_buff = NULL;
		rcv_buff = new char[rcv_buff_size];
		AVector *avector;

		for(np = 1; np < num_processors; np++)
		{
			avector = NULL;
			MPI_Recv(rcv_buff, rcv_buff_size, MPI_BYTE,np,0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
			UnpackSample(avector, rcv_buff);
			
		  // add up all the weight vectors
			vector<Feature>::const_iterator it1 = avector->features.begin();
			while(it1 != avector->features.end())
			{
				weight_vector.features[(it1->id)-1].id = it1->id;
				weight_vector.features[(it1->id)-1].weight = weight_vector.features[(it1->id)-1].weight + it1->weight;
				it1++;
			}

		}
		// add local machine weight vector finally
		vector<Feature>::const_iterator it3 = weight_vector.features.begin();
		while(it3 != weight_vector.features.end())
		{
			weight_vector.features[(it3->id)-1].id = it3->id;
			weight_vector.features[(it3->id)-1].weight =reg_prmtr * (weight_vector.features[(it3->id)-1].weight + it3->weight);
			it3++;
		}

		vector<Feature>::const_iterator it2 = weight_vector.features.begin();

	}

}

// training by normal perceptron algorithm
void train_perceptron(void)
{
	double wt_x;
	unsigned i, epoch;
	int yicap ;
	int mistakes = 0;
	
	for(epoch = 1; epoch <= MAX_EPOCH; epoch++)
	{
		mistakes = 0;
	
		MPI_Barrier(MPI_COMM_WORLD);

		for(i = 0; i < train_samples_.size(); i++) // only the samples assigned to it
		{
			double actual_label = (GetLocalSample(i)->label > 1) ? 1 : -1;

			wt_x = GetInnerProduct(*GetLocalSample(i),weight_vector);

			if (wt_x >= 0)
				yicap = 1;
			else
				yicap = -1;

			if (actual_label * yicap < 0)
			{
				Sample *sample = ScalarMultiplyVector(*GetLocalSample(i), actual_label);
  	  	AddToW(*sample);
				mistakes = mistakes+1;	
			}
		}

		MPI_Barrier(MPI_COMM_WORLD);

		// share at end of each epoch		
		recv_weight_vector_from_all();
		send_weight_vector_to_all();

		test_perceptron();

	}
}
// training by MIRA algorithm
void train_mira(void)
{
	double wt_x, yi_wt_xi,tow, norm_xisqr, temp;
	unsigned i, epoch;
	int yicap, mistakes = 0;
		
		for(epoch = 1; epoch <= MAX_EPOCH; epoch++)
		{
			mistakes = 0;
			for(i = 0; i < train_samples_.size(); i++) // only the samples assigned to it
			{
				int actual_label = (GetLocalSample(i)->label > 1) ? 1 : -1;
				wt_x = GetInnerProduct(*GetLocalSample(i),weight_vector);

				if (wt_x >= 0)
					yicap = 1;
				else
					yicap = -1;


				// calculating tow
				yi_wt_xi = actual_label * wt_x;

				if(yi_wt_xi >= 0)
					tow = 0;
				else
				{
					norm_xisqr = GetLocalSample(i)->two_norm_sq;
					
					temp = yi_wt_xi/norm_xisqr;
					if(temp <= -1)
						tow = 1;
					else
						tow = -temp;
				}
				// updating weight vector
				if ((actual_label * yicap) < 0)
				{
					
					Sample *sample = ScalarMultiplyVector(*GetLocalSample(i),(tow * actual_label));
					AddToW(*sample);
					mistakes = mistakes + 1;
				}

			}
	
			MPI_Barrier(MPI_COMM_WORLD);
			recv_weight_vector_from_all();
			send_weight_vector_to_all();

			test_perceptron();
		}

}

void train_opa(void)
{

	double wt_x, m_yi_wt_xi,tow, norm_xisqr;
	unsigned i, epoch;
	int yicap,mistakes,actual_label;


		for(epoch = 1; epoch <= MAX_EPOCH; epoch++)
		{
			mistakes = 0;
			for(i = 0; i < train_samples_.size(); i++) // only the samples assigned to it
			{

				actual_label = (GetLocalSample(i)->label > 1) ? 1 : -1;
				wt_x = GetInnerProduct(*GetLocalSample(i),weight_vector);

				if (wt_x >= 0)
					yicap = 1;
				else
					yicap = -1;


				// calculating tow
				m_yi_wt_xi = 1 - (actual_label * wt_x);
				norm_xisqr = GetLocalSample(i)->two_norm_sq;

				if(m_yi_wt_xi < 0)
					tow = 0;
				else
					tow = m_yi_wt_xi / norm_xisqr;

				// updating weight vector
				if ((actual_label * yicap) < 0)
				{
					Sample *sample = ScalarMultiplyVector(*GetLocalSample(i),(tow * actual_label));
					AddToW(*sample);
					mistakes = mistakes + 1;
				}

			}
		MPI_Barrier(MPI_COMM_WORLD);
		recv_weight_vector_from_all();
		send_weight_vector_to_all();
		test_perceptron();

		}

}

// read test data file
bool read_test_data(void)
{
	unsigned i;

	string test_file_path = "/home/rajmohan.c/project/perceptron/datasets/covtype/covtype.libsvm.binary.scale/covtype.libsvm_test.binary.scale";

	const char *t_filename = test_file_path.c_str();

	File* file = File::Open(t_filename, "r");
	if (file == NULL)
	{
	    cerr << "Cannot find file " << t_filename << endl;
	    MPI_Finalize();
	    return false;
	}

	string t_line;
	int num_test_local_pos = 0;
	int num_test_local_neg = 0;

	while (file->ReadLine(&t_line))
	{
		// If the sample should be assigned to this processor
		if (num_test_total_ % num_processors == myid)
		{
		  int label = 0;
		  const char* start = t_line.c_str();
		  // Extracts the sample's class label
		  if (!SplitOneIntToken(&start, " ", &label))
		  {
			cerr << "Error parsing line: " << num_test_total_ + 1 << endl;
			return false;
		  }

		  // Gets the local number of positive and negative samples
		  if (label == 1)
		  {
			++num_test_local_neg;
		  }
		  else if (label == 2)
		  {
			++num_test_local_pos;
		  } else
		  {
			cerr << "Unknow label in line: " << num_test_total_ + 1 << label;
			return false;
		  }

		  // Creates a "Sample" and add to the end of samples_
		  test_samples_.push_back(Sample());
		  Sample& sample = test_samples_.back();
		  sample.label = label;
		  sample.id = num_test_total_;  // Currently num_total_ == sample id
		  sample.two_norm_sq = 0.0;

		  // Extracts the sample's features
		  vector<pair<string, string> > t_kv_pairs;
		  SplitStringIntoKeyValuePairs(string(start), ":", " ", &t_kv_pairs);
		  vector<pair<string, string> >::const_iterator pair_iter;
		  for (pair_iter = t_kv_pairs.begin(); pair_iter != t_kv_pairs.end();++pair_iter)
		  {
				Feature feature;
				feature.id = atoi(pair_iter->first.c_str());
				feature.weight = atof(pair_iter->second.c_str());
				sample.features.push_back(feature);
				sample.two_norm_sq += feature.weight * feature.weight;
		  }
		}
		++num_test_total_;
	}
	file->Close();
	delete file;


  // Get the global number of positive and negative samples
  int local[2];
  int global[2];
  local[0] = num_test_local_pos;
  local[1] = num_test_local_neg;
  memset(global, 0, sizeof(global[0] * 2));

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Allreduce(local, global, 2, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  num_test_pos_ = global[0];
  num_test_neg_ = global[1];

/*  if (myid == 0)
  {
	  cout << "Total: " << num_test_total_
				<< "  Positive: " << num_test_pos_
				<< "  Negative: " << num_test_neg_ << endl;
  }
*/
  return true;
}

void test_perceptron(void)
{
	int i;
	double wt_x = 0;
	int test_mistakes = 0;
	char actual_label, predicted_label;

	  for(i = 0; i < test_samples_.size(); i++)
	  {
	    	actual_label = ((GetLocalTestSample(i)->label) > 1) ? 1 : -1;
	    	wt_x = GetInnerProduct(*GetLocalTestSample(i),weight_vector);

    		if(wt_x < 0)
					predicted_label = -1;
				else
					predicted_label = 1;

			if(actual_label != predicted_label)
				test_mistakes = test_mistakes + 1;

	  }
	
	double test_error = ((double)test_mistakes/test_samples_.size()) * 100.0;
	cout <<myid<< ": Test Error % =" << test_error<<endl;
}


double GetInnerProduct(const Sample& a, AVector w)
{
  double norm = 0.0;

  vector<Feature>::const_iterator it1 = a.features.begin();
  vector<Feature>::const_iterator it2 = w.features.begin();

  while (it1 != a.features.end() && it2 != w.features.end())
  {
    if ((it1->id) == (it2->id))
	{
      norm += ((it1->weight) * (it2->weight));
      ++it1;
      ++it2;
    }
	else if ((it1->id) < (it2->id))
	{
      ++it1;
    }
	else
	{
      ++it2;
    }
  }
  return norm;
}

Sample* ScalarMultiplyVector(const Sample& a,double scalar)
{
  Sample* sample = new Sample();

  vector<Feature>::const_iterator it1 = a.features.begin();
  sample->two_norm_sq = 0.0;

  while (it1 != a.features.end() )
  {
      Feature feature;
      feature.id = it1->id;
      feature.weight = (it1->weight) * scalar;
      sample->features.push_back(feature);
      sample->two_norm_sq += (feature.weight * feature.weight);

      ++it1;
  }
  return sample;
  
}

void AddToW(const Sample &sample)
{
	vector<Feature>::const_iterator it1 = sample.features.begin();
	vector<Feature>::const_iterator it2 = weight_vector.features.begin();


	while(it1!= sample.features.end())
	{
		//cout << it1->id << ":" << it1->weight<<" " ;
  		weight_vector.features[(it1->id)-1].id = it1->id;
  		weight_vector.features[(it1->id)-1].weight = weight_vector.features[(it1->id)-1].weight + it1->weight;
		it1++;
	}

}

const Sample* GetLocalSample(int local_row_index)
{
  // If local_row_index points to a illegal position, returns NULL.
  if (local_row_index < 0 || local_row_index >= (int)train_samples_.size())
  {
    return NULL;
  }

  // Otherwise returns a const pointer to the sample.
  return &(train_samples_[local_row_index]);
}

const Sample* GetLocalTestSample(int local_row_index)
{
  // If local_row_index points to a illegal position, returns NULL.
  if (local_row_index < 0 || local_row_index >= (int)test_samples_.size())
  {
    return NULL;
  }

  // Otherwise returns a const pointer to the sample.
  return &(test_samples_[local_row_index]);
}

size_t GetPackSize(const AVector & sample)
{
  // Size of the first three data members of Sample
  int size_buffer = 0;

  // Size of num_features. We need to encode it to indicate how many features there are in the memory block.
  size_t num_features = sample.features.size();
  size_buffer += sizeof(num_features);

  // Size of sample.features, which is (sizeof(id) + sizeof(weight) * num_features. We do not use
  // sizeof(Feature)since the two are not always equal. See the definition of Feature in the header file.
  size_buffer += (sizeof(sample.features[0].id) +
                  sizeof(sample.features[0].weight)) * num_features;

  return size_buffer;
}

size_t PackSample(char *&buffer, const AVector &sample)
{
  // buffer should be a pre-allocated memory block, with proper block size.
  // If buffer is not allocated, say "buffer == NULL", then allocates memory
  if (buffer == NULL) {
    size_t size_buffer = GetPackSize(sample);
    buffer = new char[size_buffer];
  }

  size_t offset = 0;

  // Encodes num_features
  size_t num_features = sample.features.size();
  memcpy(buffer + offset, &num_features, sizeof(num_features));
  offset += sizeof(num_features);

  // Encodes sample.features
  for (size_t i = 0; i < num_features; ++i)
  {
    // Encodes one feature
    memcpy(buffer + offset, &(sample.features[i].id), sizeof(sample.features[i].id));
    offset += sizeof(sample.features[i].id);
    memcpy(buffer + offset,&(sample.features[i].weight),sizeof(sample.features[i].weight));
    offset += sizeof(sample.features[i].weight);
  }

  return offset;
}

size_t UnpackSample(AVector *&sample, const char *buffer)
{
  size_t offset = 0;
  if (sample == NULL)
	  sample = new AVector();

  // Decodes num_features
  size_t num_features;
  memcpy(&num_features, buffer + offset, sizeof(num_features));
  offset += sizeof(num_features);

  // Decodes sample.features
  for (size_t i = 0; i < num_features; ++i)
  {
    Feature feature;

    // Decodes one feature
    memcpy(&(feature.id), buffer + offset, sizeof(feature.id));
    offset += sizeof(feature.id);
    memcpy(&(feature.weight), buffer + offset, sizeof(feature.weight));
    offset += sizeof(feature.weight);

    sample->features.push_back(feature);
  }

  return offset;
}

