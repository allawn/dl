#include <tensorflow/core/public/session.h>
#include <tensorflow/core/protobuf/meta_graph.pb.h>

using namespace std;
using namespace tensorflow;

...
// set up your input paths
const string pathToGraph = "models/my-model.meta"
const string checkpointPath = "models/my-model";
...

auto session = NewSession(SessionOptions());
if (session == nullptr) {
    throw runtime_error("Could not create Tensorflow session.");
}

Status status;

// Read in the protobuf graph we exported
MetaGraphDef graph_def;
status = ReadBinaryProto(Env::Default(), pathToGraph, &graph_def);
if (!status.ok()) {
    throw runtime_error("Error reading graph definition from " + pathToGraph + ": " + status.ToString());
}

// Add the graph to the session
status = session->Create(graph_def.graph_def());
if (!status.ok()) {
    throw runtime_error("Error creating graph: " + status.ToString());
}

// Read weights from the saved checkpoint
Tensor checkpointPathTensor(DT_STRING, TensorShape());
checkpointPathTensor.scalar<std::string>()() = checkpointPath;
status = session->Run(
        {{ graph_def.saver_def().filename_tensor_name(), checkpointPathTensor },},
        {},
        {graph_def.saver_def().restore_op_name()},
        nullptr);
if (!status.ok()) {
    throw runtime_error("Error loading checkpoint from " + checkpointPath + ": " + status.ToString());
}

// and run the inference to your liking
auto feedDict = ...
auto outputOps = ...
std::vector<tensorflow::Tensor> outputTensors;
status = session->Run(feedDict, outputOps, {}, &outputTensors);