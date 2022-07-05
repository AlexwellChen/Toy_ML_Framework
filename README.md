# Toy ML Framework

This Framework comes from CSE599W Assignment1, which I'm aiming to build a Machine Learning System with Automatic differentiation. This framework will contain following modules: Automatic differentiation, User interface, CUDA Ops and etc.

### Automatic Differentiation

The AD (Automatic differentiation) is the core module of the whole system. With AD there is no need to take care of the differentiation manually, which will be done by the system. All we need to do is impliment the Ops' compute and gradient function.

*Computation graph*

The basis of automatic differentiation is the computation graph, which can be divided into Forward Mode and Reverse Mode according to the different graph traversal methods.

Forward Mode is to traverse the entire computation graph whenever we have an independent variable input in order to compute its differentiation, which generates a lot of unnecessary node visits.

Reverse Mode is to traverse in reverse topological order from the output node, so that through a reverse traversal, we record the gradient of the precursor node of each node, and sum up all the gradients coming from the back after reaching each node to get the total gradient size, and then we continue to calculate the gradient components of the precursor node according to this gradient, and finally get the gradient of the output y with respect to each input variable.

This framework uses Reverse Mode and generates a static computational graph.

*Graph Node*

Graph node is defined as follow.

```C++
class Node {
public:
    Node(){
        this->const_attr = 0.0;
    }
    vector<Node> input;
    Op *op;
    float const_attr;
    string name;
    int hash_code;
    bool isPlaceHolder;

    virtual Node operator+(Node &nodeB);

    virtual Node operator*(Node &nodeB);

    ~Node() = default;
};
```

All nodes have its own operator, which indicates that what type of computation will be done in current node. The Operator is defined as follow.

```C++
class Op {
public:
    Op() = default;

    virtual Node getNewNode();
		// T could be vector or matrix
    virtual vector<T> compute(Node &node, vector<T> &input_vals);

    virtual vector<Node> gradient(Node &node, Node &output_gradient);

    ~Op() = default;
};
```

Each operator needs to impliment three function. We need getNewNode to generate a node which corresponds current operator, and use compute and gradient to represent the math computation.

*Constructing back-propagation graph of gradients*

The example is y = x1 * x2 + x3.

The construction of the gradient backpropagation graph is done through the gradients function, the implementation of which can be found in autodiff.cpp. The gradients function has two inputs: one is the output node, which is our y, and the other is the list of input nodes, which is our [x1, x2, x3]. Since we need to back-propagate, we traverse the nodes using a reverse topological sort, ensuring that when we visit a node, we have visited all the nodes after it (i.e. we have gotten all its gradient components).

The first thing we can know is that the derivative of the output node y with respect to the expression x1*x2+x3 is 1. We store this value in the grad_map corresponding to y.

Then we topologically sort the computational map and reverse to get the node access order and then we can construct the back propagation map.

For each node we follow the following process.

1. look up the table, get the gradient passed by all output nodes of the current node and sum up (each gradient component is a Node node, as mentioned above, the gradient specific value calculation still needs to rely on the compute function to complete, so the sum up here is to build the expression node of the gradient sum).

2. save the gradient sum of the current node.

3. Use the gradient function of the current node operator to build the gradient component expression nodes of all input Nodes.

4. Save the gradient component nodes of each input Node to node_to_output_grads_list and wait for the summation.

After getting the gradient expression nodes for each node, we can pass the gradient expression nodes back in a certain order and wait to put them into the executor for execution.

*The Executor*

The constructor input to Executor is a list [y, x1_grad, x2_grad, x3_grad], where y contains the expressions we need to evaluate, followed by the gradient value of each variable with respect to y.

Executor has a run function to execute the entire computational graph as well as the gradient back-propagation graph.

We already have all the nodes when we construct the executor instance, we just need to sort them topologically to get the node traversal list. Since we have already constructed the gradient backpropagation graph, we are actually combining the two graphs into one for topological sorting.

In the forward propagation, we calculate the value of each node, and then we enter the backpropagation graph and still use the compute function to calculate the value of the current node according to the corresponding rules, at this time we calculate the gradient value of the corresponding gradient expression node. Finally, we return the values of the nodes [y, x1_grad, x2_grad, x3_grad] that we need to the main function.

### Todo List

* Tensorflow/Keras like User Interface
* CUDA Operators
* Convolution Operators
* Computation graph serialization and deserialization
