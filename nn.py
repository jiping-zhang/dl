import numpy as np
import json
import random
from abc import abstractmethod
from typing import *

E = 2.718281828

STR_TO_CLASS = {} # every none-abstract class should register itself in this dict , or it can't be saved
CLASS_TO_STR = {}


def model_to_json_obj(model)->dict:
    return {'type': CLASS_TO_STR[type(model)], 'content': model.jsonify()}


def model_to_str(model)->str:
    return json.dumps(model_to_json_obj(model))


def save(model, path: str):
    with open(path, 'w') as fout:
        fout.write(model_to_str(model))


def json_obj_to_model(obj: dict):
    clazz = STR_TO_CLASS[obj['type']]
    instance = clazz()
    instance.load(obj['content'])
    return instance


def str_to_model(string:str):
    return json_obj_to_model(json.loads(string))


def load(path: str):
    with open(path, 'r') as fin:
        return str_to_model(fin.readline())


# Instances of this class will only be used in Momentum optimize method
class Gradient:
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __mul__(self, k: float):
        raise NotImplementedError

    def __rmul__(self, k: float):
        return self.__mul__(k)

    @abstractmethod
    def __add__(self, another):
        raise NotImplementedError


class Layer:
    # Any class extends Layer has to provide a no-arg constructor ,
    #           or there is going to be trouble while loading 
    def __init__(self) -> None:
        pass

    def __call__(self, input: np.ndarray) -> np.ndarray:
        return self.forward(input)
    
    @abstractmethod
    def forward(self, input: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def step(self, lr: float) -> None:
        raise NotImplementedError

    # some layers , such as dropout , has difference between training and evaluating
    # to be added , layers needn't calculate gradient while evaluating
    @abstractmethod
    def train(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> None:
        raise NotImplementedError

    # set L2 normalization factor
    @abstractmethod
    def set_weight_decay(self, k: float) -> None:
        raise NotImplementedError

    # following three methods will only be used in Momentum optimize method
    @abstractmethod
    def get_self_gradient(self) -> Gradient:
        raise NotImplementedError

    @abstractmethod
    def add_from_gradient(self, gradient: Gradient) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_from_gradient(self, gradient: Gradient) -> None:
        raise NotImplementedError

    # following two methods are used to save model to file and load model from file
    # jsonify() : convert self to an object that can be dumped to a json-object .
    # built-in types , including [int,float,str,list,dict] can be dumped .
    @abstractmethod
    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        raise NotImplementedError

    @abstractmethod
    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        raise NotImplementedError


class GradientGroup(Gradient):
    def __init__(self, gradients: List[Gradient]) -> None:
        super().__init__()
        self.gradients = gradients

    def __add__(self, another):
        return GradientGroup(list(map(lambda tup:tup[0]+tup[1],zip(self.gradients,another.gradients))))

    def __mul__(self, k: float):
        return GradientGroup([grad * k for grad in self.gradients])

    def __getitem__(self, index: int) -> Gradient:
        return self.gradients[index]


class Sequential(Layer):
    def __init__(self, layers: List[Layer] = []) -> None:
        super().__init__()
        self.layers = layers

    def train(self) -> None:
        for layer in self.layers:
            layer.train()

    def eval(self) -> None:
        for layer in self.layers:
            layer.eval()

    def forward(self, input: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for i in range(len(self.layers)):
            grad = (self.layers[len(self.layers)-i-1]).backward(grad)
        return grad

    def step(self, lr: float) -> None:
        for layer in self.layers:
            layer.step(lr)

    def set_weight_decay(self, k: float) -> None:
        for layer in self.layers:
            layer.set_weight_decay(k)

    def get_self_gradient(self) -> GradientGroup:
        return GradientGroup([layer.get_self_gradient() for layer in self.layers])

    def add_from_gradient(self, gradient: GradientGroup) -> None:
        for i in range(len(self.layers)):
            self.layers[i].add_from_gradient(gradient[i])

    def load_from_gradient(self, gradient: GradientGroup) -> None:
        for i in range(len(self.layers)):
            self.layers[i].load_from_gradient(gradient[i])

    def __getitem__(self, index: int) -> Layer:
        return self.layers[index]

    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        return [model_to_json_obj(layer) for layer in self.layers]

    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        self.layers = [json_obj_to_model(layer_obj) for layer_obj in obj]


class LinearLayerGradient(Gradient):
    def __init__(self, weight_grad: np.ndarray, bias_grad: np.ndarray) -> None:
        super().__init__()
        self.weight_grad = weight_grad
        self.bias_grad = bias_grad

    def __add__(self, another):
        return LinearLayerGradient(self.weight_grad+another.weight_grad, self.bias_grad+another.bias_grad)

    def __mul__(self, k: float):
        return LinearLayerGradient(k*self.weight_grad, k*self.bias_grad)


class Linear(Layer):
    # def __init__(self, n_in: int = 1, n_out: int = 1,*_,weight_loc:float=0.0,weight_scale:float=1.0,bias_loc:float=0.0,bias_scale:float=1.0) -> None:
    def __init__(self, n_in: int = 1, n_out: int = 1) -> None:
        super().__init__()
        self.weight = np.random.normal(loc=0.0, scale=1.0, size=(n_in, n_out))
        self.bias = np.random.normal(loc=0.0, scale=1.0, size=(n_out))
        self.weight_grad = np.array([[0.0]*n_in]*n_out,dtype=float)
        self.bias_grad = np.array([0.0]*n_out,dtype=float)
        self.input = None
        self.__train = True
        self.__decay_factor = 0.0 # L2 normalization factor

    def train(self) -> None:
        self.__train = True

    def eval(self) -> None:
        self.__train = False

    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.__train:
            self.input = input
        return np.dot(input, self.weight)+self.bias

    def backward(self, grad: np.ndarray) -> np.ndarray:
        batch_size = len(self.input)
        batch_size_div = 1/batch_size
        self.weight_grad = batch_size_div * np.dot(np.transpose(self.input), grad) + self.__decay_factor * self.weight
        self.bias_grad = batch_size_div*np.sum(grad, 0)# L2 regulation needn't be imposed on bias
        return np.dot(grad, np.transpose(self.weight))

    def step(self, lr: float) -> None:
        if not self.__train:
            print('Warning : Optimizing a Linear Layer while it is in evaluating mode')
        self.weight += (-lr)*self.weight_grad
        self.bias += (-lr)*self.bias_grad

    def set_weight_decay(self, k: float) -> None:
        self.__decay_factor = float(k)

    def get_self_gradient(self) -> LinearLayerGradient:
        return LinearLayerGradient(self.weight_grad.copy(), self.bias_grad.copy())

    def add_from_gradient(self, gradient: LinearLayerGradient) -> None:
        self.weight_grad += gradient.weight_grad
        self.bias_grad += gradient.bias_grad

    def load_from_gradient(self, gradient: LinearLayerGradient) -> None:
        self.weight_grad = gradient.weight_grad
        self.bias_grad = gradient.bias_grad

    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        return {
            'weight': self.weight.tolist(),  # Convert np.ndarray to built-in list , ndarray can't be jsonified 
            'bias': self.bias.tolist()
        }

    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        self.weight_grad = 0.0
        self.bias_grad = 0.0
        self.weight = np.array(obj['weight'],dtype = float)
        self.bias = np.array(obj['bias'],dtype = float)


class NoneOptimLayers(Layer):
    def __init__(self) -> None:
        super().__init__()

    def step(self, lr: float) -> None:
        pass

    def set_weight_decay(self, k: float) -> None:
        pass

    # Since none-optim layers don't need to be optimized , they don't need to output their gradient 
    # It can't return None here , or there's going to be trouble when adding two grads
    def get_self_gradient(self) -> Gradient:
        return 0.0

    def add_from_gradient(self, gradient: Gradient) -> None:
        pass

    def load_from_gradient(self, gradient: Gradient) -> None:
        pass


class Activation(NoneOptimLayers):
    def __init__(self) -> None:
        super().__init__()

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        return []

    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        pass


class ReLU(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.output = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        output = np.maximum(input, 0.0)
        self.output = output.copy()
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return (self.output > 0)*grad


#  y =     x , when x >  0
#      k * x , when x <= 0
class LeakyReLU(Activation):
    def __init__(self, k:float=0.5) -> None:
        super().__init__()
        assert 0.0 < k < 1.0
        self.k = k
        self.mask = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.mask = (input > 0)+self.k * (input <= 0) # True -> 1 , False -> 0  when bool multiplies float 
        return self.mask*input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.mask*grad

    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        return str(self.k)

    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        self.k = float(obj)


# y = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) = 1 - 2 / (1 + exp(2x))
# dy / dx = 1 / ((cosh(x)) ^ 2) = 4 / (exp(2x) + 2 + exp(-2x))
class Tanh(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.exp_2x = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        exp_2x = E**(2*input)
        self.exp_2x = exp_2x
        return 1.0-2.0/(exp_2x+1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return (4.0/(self.exp_2x+2.0+(1.0/self.exp_2x)))*grad


# y = 1 / (1 + exp(-x))
# dy / dx = y * (1 - y)
class Sigmoid(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.output = None

    def __sigmoid(self, x):
        return 1.0/(1.0+E**(-x))

    def __y2grad(self, y):
        return y*(1-y)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.shape = input.shape
        output = self.__sigmoid(input)
        self.output = output.copy()
        return output

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad * self.__y2grad(self.output)


class Dropout(NoneOptimLayers):
    def __init__(self, drop_out_rate: float = 0.5) -> None:
        assert 0.0 < drop_out_rate < 1.0
        self.drop_out_rate = drop_out_rate
        self.save_rate_div = 1.0/(1.0-drop_out_rate)
        self.mask = None
        self.__train = True

    def train(self) -> None:
        self.__train = True

    def eval(self) -> None:
        self.__train = False

    def forward(self, input: np.ndarray) -> np.ndarray:
        if self.__train:
            self.mask = np.array([random.random() >= self.drop_out_rate for _ in range(len(input[0]))], dtype=float)
            return input*self.mask*self.save_rate_div
        else:
            return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self.__train:
            return grad*self.mask*self.drop_out_rate
        else:
            return grad

    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        return str(self.drop_out_rate)

    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        self.drop_out_rate = float(obj)
        self.save_rate_div = 1.0/(1.0-self.drop_out_rate)


EPSILON = 1e-8

# mean -> 0.0 , var -> 1.0
class Normalization(Activation):
    def __init__(self) -> None:
        super().__init__()
        self.var_div = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        var = np.var(input,1)+EPSILON # in case of 0
        self.var_div = 1.0/var
        self.var_div.shape = -1,1
        avg = np.mean(input,1)
        avg.shape = -1,1
        return (input-avg)/self.var_div

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return grad*self.var_div


class LinearShiftGradient(Gradient):
    def __init__(self,kg:float,bg:float) -> None:
        super().__init__()
        self.kg = kg
        self.bg= bg

    def __add__(self, another):
        return LinearShiftGradient(self.kg+another.kg,self.bg+another.bg)

    def __mul__(self, k: float):
        return LinearShiftGradient(self.kg*k,self.bg*k)

# output = k * input + b
# k is not a matrix , this layer applies (kx+b) to every feature
class LinearShift(Layer):
    def __init__(self) -> None:
        self.__k = np.random.random()
        self.__k_grad = 0.0
        self.__b = np.random.random()
        self.__b_grad = 0.0
        self.output = None
        self.__batch_size_div = 0.0

    def train(self) -> None:
        pass

    def eval(self) -> None:
        pass

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.__batch_size_div = 1.0 / len(input)
        self.output = self.__k*input + self.__b
        return self.output.copy()

    def backward(self, grad: np.ndarray) -> np.ndarray:
        self.__k_grad = self.__batch_size_div*np.sum(grad*self.output)
        self.__b_grad = self.__batch_size_div*np.sum(grad)
        return self.__k * grad

    def step(self, lr: float) -> None:
        self.__k -= lr*self.__k_grad
        self.__b -= lr*self.__b_grad

    def set_weight_decay(self, k: float) -> None:
        pass

    def get_self_gradient(self) -> LinearShiftGradient:
        return LinearShiftGradient(self.__k_grad,self.__b_grad)

    def add_from_gradient(self, gradient: LinearShiftGradient) -> None:
        self.__k_grad += gradient.kg
        self.__b_grad += gradient.bg

    def load_from_gradient(self, gradient: LinearShiftGradient) -> None:
        self.__k_grad = gradient.kg
        self.__b_grad = gradient.bg

    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        return {
            "k":self.__k,
            "b":self.__b
        }

    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        self.__k = obj["k"]
        self.__b = obj["b"]


class BatchNormalization(Layer):
    def __init__(self) -> None:
        super().__init__()
        self.normalize = Normalization()
        self.shift = LinearShift()

    def train(self) -> None:
        self.shift.train()

    def eval(self) -> None:
        self.shift.eval()

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.shift.forward(self.normalize.forward(input))

    def backward(self, grad: np.ndarray) -> np.ndarray:
        return self.normalize.backward(self.shift.backward(grad))

    def step(self, lr: float) -> None:
        self.shift.step(lr)

    def set_weight_decay(self, k: float) -> None:
        pass

    def get_self_gradient(self) -> Gradient:
        return self.shift.get_self_gradient()

    def add_from_gradient(self, gradient: Gradient) -> None:
        self.shift.add_from_gradient(gradient)

    def load_from_gradient(self, gradient: Gradient) -> None:
        self.shift.load_from_gradient(gradient)

    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        return self.shift.jsonify()

    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        self.shift.load(obj)


class Reshape(Activation):
    def __init__(self, in_shape: Union[List[int], Tuple[int]] = [-1], out_shape: Union[List[int], Tuple[int]] = [-1]) -> None:
        super().__init__()
        self.__in_shape = list(in_shape)
        if self.__in_shape[0] != -1:
            self.__in_shape.insert(0, -1)
        self.__out_shape = list(out_shape)
        if self.__out_shape[0] != -1:
            self.__out_shape.insert(0, -1)

    def forward(self, input: np.ndarray) -> np.ndarray:
        input.shape = self.__out_shape
        return input

    def backward(self, grad: np.ndarray) -> np.ndarray:
        grad.shape = self.__in_shape
        return grad

    def jsonify(self) -> Union[list, dict, tuple, str, float, int]:
        return {'in': self.__in_shape, 'out': self.__out_shape}

    def load(self, obj: Union[list, dict, tuple, str, float, int]) -> None:
        self.__in_shape = obj['in']
        self.__out_shape = obj['out']


CLASS_TO_STR = {
    Linear: 'Lin',
    Sigmoid: 'Sgm',
    Tanh: 'Tanh',
    ReLU: 'ReLU',
    LeakyReLU: 'LReLU',
    Sequential: 'Seq',
    Dropout: 'Drop',
    Reshape: 'Reshape',
    BatchNormalization:'BN',
}

STR_TO_CLASS = {CLASS_TO_STR[key]: key for key in CLASS_TO_STR}


class LossFunc:
    def __init__(self) -> None:
        pass

    def __call__(self, output: np.ndarray, label: np.ndarray) -> float:
        return self.calculate_loss(output=output, label=label)

    @abstractmethod
    def calculate_loss(self, output: np.ndarray, label: np.ndarray) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_grad(self) -> np.ndarray:
        raise NotImplementedError


# loss = 1/2 * ((y_out-y_label)^2)
# grad = dloss / dy_out = y_out - y_label
class MeanSquareError(LossFunc):
    def __init__(self) -> None:
        super().__init__()
        self.grad = None

    def calculate_loss(self, output: np.ndarray, label: np.ndarray) -> float:
        self.grad = output-label
        return 0.5*np.mean(self.grad**2)

    def get_grad(self) -> np.ndarray:
        return self.grad
    

# loss = absolute(out-label)    
class AbsoluteError(LossFunc):
    def __init__(self) -> None:
        super().__init__()
        self.grad = None
        
    def calculate_loss(self, output: np.ndarray, label: np.ndarray) -> float:
        self.grad = np.array(output>label,dtype=float)
        return self.grad * (output-label)
    
    def get_grad(self) -> np.ndarray:
        return self.grad 


class CrossEntropy(LossFunc):
    def __init__(self) -> None:
        super().__init__()
        self.grad = None
        self.batch_output = None
        self.batch_label = None
        self.batch_size = None

    def calculate_loss(self, output: np.ndarray, label: np.ndarray) -> float:
        self.batch_output = output
        self.batch_label = label
        self.batch_size = len(output)
        return None

    def get_grad(self) -> np.ndarray:
        output_len = len(self.batch_output[0])
        exp_output = E ** self.batch_output
        sum_exp = np.sum(exp_output, 1)
        sum_exp.shape = -1, 1
        probability = exp_output/sum_exp
        grad_loss_on_p = ((1-self.batch_label)/(1-probability) -self.batch_label/probability)/output_len
        grad_p_on_output = exp_output*(sum_exp-exp_output) / sum_exp**2
        return grad_loss_on_p*grad_p_on_output
