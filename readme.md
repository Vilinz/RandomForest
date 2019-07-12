**一、**    **题目要求**

使用随机森林算法对给出的数据集完成回归任务，要求：

\1.    对测试集的数据进行预测。

\2.    能够并行化处理。

\3.    优化算法，使得cache的换页变得尽可能少。

 

**二、**    **随机森林简介**

随机森林是最简单的聚类算法之一，算法的过程是群体智慧的体现，简单地说，随机森林就是用随机的方式建立一个森林，森林里面有很多的决策树组成，随机森林的每一棵决策树之间是没有关联的。随机森林有两类基本任务，分别是分类和回归，对于分类来说，将数据输入到随机森林的每一棵树当中，每棵树会产生一个输出，我们取被选择最多的作为我们预测的结果；对于回归任务来说，对于随机森林中的每棵树的输出求和然后取平均值作为预测的结果。

 

**三、**    **技术栈选择**

本次给出的任务是一个回归的任务，所以我们要写的是回归树，语言的选择是python，虽然python代码执行的效率比较低，但python对于数据的读取，矩阵方面的计算会有优化，例如numpy这个包对矩阵的四则运行有一定的优化。

另外是并行化选择的技术，在python当中，对于并行化的处理有两种方式，分别是多线程和多进程，但注意两者的区别与选择，两者的选择是有原则的：

\1.    多线程

适合用于IO密集型的运算，比如文件读取等。

\2.    多进程

适合用于CPU密集型的运算，比如计算等。

上面选择的理由也是比较明显的，对于IO密集型操作，大部分消耗时间其实是等待时间，等待期间不占用CPU资源，在等待时间中CPU是不需要工作的，因为此时CPU的时间片会被别的程序占用，那你在此期间提供多个CPU资源也是利用不上的，相反对于CPU密集型代码，多个CPU干活肯定比一个CPU快很多。

对照我们的项目，选择分割点的过程全是计算密集型的，所以我选择了多进程来进行并行化。

 

**四、**    **算法流程**

在这个算法需要用到bootstrap的思想，在给出详细算法之前，我们先来了解一下bootstrap是什么，bootstrap就是从原始数据当中选取子集的思想，采用的是有放回的方式选取数据集，为了保证随机森林当中的每棵树都是相对独立的，算法需要为每棵树提供一个原始数据集的子集。

基础版本的算法如下，

l  For each tree in the forest(from b = 1 to B)

n  Using bootstrap to choose a sub dataset z

n  Repeat

u  Select m features at random from p features(all feature)

u  Pick the best variable among m features(from I = 1 to m)

l  Get all different values of one feature

l  If the different of the features is to similar, delete the value

u  Spilt the node into two daughter nodes