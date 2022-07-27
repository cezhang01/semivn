# SemiVN
Source code and datasets of ECML/PKDD-21 paper, [Semi-Supervised Semantic Visualization for Networked Documents](https://www.dropbox.com/s/9958fbvxn7y5l80/ecmlpkdd21b.pdf?dl=0), by [Delvin Ce Zhang](http://www.delvincezhang.com) and [Hady W. Lauw](http://www.hadylauw.com).

SemiVN is a model that can i) extract latent topics from a collection of documents, and ii) visualize documents, topics, and labels.

![](/paper/interaction.jpg)

## Implementation Environment
- numpy == 1.17.4
- tensorflow == 1.9.0
- networkx == 2.4
- matplotlib == 3.0.3
- wordcloud == 1.6.0
- sklearn == 0.21.3
- scipy == 1.3.1

## Run
After convergence, the program will show visualization plot. If the dataset is coronavirus, users can interact with the plot. Right click topics and labels to show word clouds. Left click documents to show specific content at control window. (Note: It is possible to show article inside the plot, but due to its long description, we show it in a separate window for clarity.) If the dataset is DS, users can only see visualization, but cannot interact, since DS dataset does not have original complete content.

`python main.py -dn coronavirus`, or `python main.py -dn ds`

### Parameter Setting
- -lr: learning rate, default = 0.1
- -ne: number of epochs for iterations, default = 300
- -dn: dataset name, ds or coronavirus
- -ra: labeling ratio of documents, default = 0.8
- -nn: number of negative samples, default = 5
- -nt: number of topics, default = 30
- -vd: dimension of visualization coordinates, default = 2
- -ms: minibatch size, 0 = batch gradient descent, other positive numbers = stochastic gradient descent, default = 128
- -l: lambda, label smoothness regularizer, default = 1
- -ii: if users want to directly call visualization of previous running results, set ii to 1; if users want to train the model and see visualization after training convergence, set ii to 0, default = 0
- -rs: random seed, we randomly generate 5 different random seeds to run experiments independently, and report both mean and standard deviation in the main paper

## Output
Results will be output to `./results` file.
- `topic_word.txt` contains #topics row, each row is a distribution over #words words
- `label_word.txt` contains #labels row, each row is a distribution over #words words
- `vertex_coor.txt` contains document coordinates, #documents rows, each row has 2 dimensions
- `topic_coor.txt` contains topic coordinates, #topics rows, each row has 2 dimensions
- `label_coor.txt` contains label coordinates, #labels rows, each row has 2 dimensions
- `topic_top_words.txt` contains top keywords of each topic, #topics rows, each row has 20 keywords
- `label_top_words.txt` contains top keywords of each label, #labels rows, each row has 20 keywords

## Reference
If you use our paper, including code and data, please cite

```
@inproceedings{semivn,
  title={Semi-supervised semantic visualization for networked documents},
  author={Zhang, Delvin Ce and Lauw, Hady W},
  booktitle={Joint European Conference on Machine Learning and Knowledge Discovery in Databases},
  pages={762--778},
  year={2021},
  organization={Springer}
}
```
