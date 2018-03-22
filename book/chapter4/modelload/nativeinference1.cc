#include <stdio.h>
#include <stdlib.h>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/common_runtime/dma_helper.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

using namespace tensorflow;
using tensorflow::Tensor;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::int32;

void LoadGraph(string deviceId, string graph_file_name, MetaGraphDef &graph_def, Session* & session) {
	    Env* env = Env::Default();
	    const char* library_filename="userlib.so";
	    void* handle=nullptr;
	    Status aa=env->LoadLibrary(library_filename,&handle);
	    LOG(ERROR)<<aa.ToString()<<endl;

	    Status load_graph_status =ReadBinaryProto(env, graph_file_name, &graph_def);
	    if (!load_graph_status.ok()) {
	        LOG(ERROR)<<"Failed to load:"<<graph_file_name<<endl;
	        exit(0);
	    }
	    SessionOptions opts;
	    //opts.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);
	    graph::SetDefaultDevice(deviceId, graph_def.mutable_graph_def());
	    opts.config.mutable_gpu_options()->set_allow_growth(true);
	    opts.config.set_allow_soft_placement(true);
	    opts.config.set_log_device_placement(false);
	    NewSession(opts, &session);
	    Status ss = session->Create(graph_def.graph_def());

	    if(ss.ok()){
	       LOG(INFO)<<"Successfully."<<endl;
	    }
	    else{
	       LOG(ERROR)<<"Failed create session."<<endl;
	       LOG(ERROR)<<ss.ToString()<<endl;
	       exit(0);
	    }
}

int main(int argc,const char **argv){
    tensorflow::MetaGraphDef graph_def;
	string graph_pb_file="user.meta";
	Session* session_;
	LoadGraph("/gpu:0", graph_pb_file,graph_def,session_);
	cout<<"-----test collection def size:"<<graph_def.collection_def_size()<<endl;

        //deal meta data
    	for (const auto& col : graph_def.collection_def()) {
    		const CollectionDef& collection = col.second;
    		string keyStr=col.first;
    		cout <<"collection key: "<<keyStr << endl;
            if(keyStr=="optname"){
    			if (collection.has_bytes_list()) {
    				for (const auto& node : collection.bytes_list().value()) {
    					opt_op_name=node;
    					cout<<"\toptName is:"<<opt_op_name<<endl;
    				}
    			}
            }
        }
        session_->Run({},{},{"initvars"},nullptr);
}