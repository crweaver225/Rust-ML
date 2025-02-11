mod neural_network;
use neural_network::*;
use tch::Tensor;

struct MyModel {
    l1: Linear,
    l2: Linear,
}

impl MyModel {
    fn new (net: &mut Network) -> MyModel {
        let l1 = Linear::new(net, 784, 128);
        let l2 = Linear::new(net, 128, 10);

        Self {
            l1: l1,
            l2: l2,
        }
    }
}

impl Compute for MyModel {
    fn forward (&self,  net: &Network, input: &Tensor) -> Tensor {
        let mut o = self.l1.forward(net, &input);
        o = o.relu();
        self.l2.forward(net, &o)
    }
}

fn main() {

    let (x, y) = load_mnist();
    let mut m = Network::new();
    let mymodel = MyModel::new(&mut m);
    train(&mut m, &x, &y, &mymodel, 10, 128, cross_entropy, 0.001);
    let out = mymodel.forward(&m, &x);
    let acc = out.accuracy_for_logits(&y);
    println!("Accuracy: {}", acc);
}