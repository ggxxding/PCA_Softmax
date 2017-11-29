/*
1.1	包的命名都使用小写的英文字母组成，每个包名称之间用点号分隔开。
1.2	全局包的名字这里约定为com.shu
2、	类、编译单元命名
2.1 命名类：
约定使用完全的英文描述符，所有单词的第一个字符大写，并且单词中大小写混合。类名应是单数形式。
示例：Kmeans， RandomForest。
2.2 命名编译单元：
编译单元在这个情况下是一个源码文件，应被赋予文件内定义的主要的类的名字。
用与类相同的名字命名文件，扩展名.java作为后缀名。示例：RandomForest.java。
成员函数的命名应采用完整的英文描述符，大小写混合使用：所有中间单词的首字母大写.
有四个函数在类中一定要有，
分别是：
用于初始化的构造函数（构造函数名与类名相同），
训练模型函数fit，
预测函数predict，
评分函数score。
其他成员函数的名字请自行起名，但第一个单词应使用一个有强烈动作色彩的动词。
4、	字段、属性命名标准
应使用完整的英文描述符来命名字段，以便使字段所表达的意思一目了然。
像数组或者是矢量这样是集合的字段，命名时应使用复数来表达它们为多值。示例：orderItems。
5、	局部变量命名标准
命名局部变量遵循与命名字段一样的规范
6、	类及成员函数参数命名标准
（1）常数参数使用希腊字母罗马音进行规范（alpha, beta, gamma, delta）.
（2）若参数需要用两个及以上单词命名，单词间用下划线”_”分割，单词首字母全小写（如fit_intercept），若第二次单词只有一个字母，则大写（如copy_X）.
（3）对于长单词（10字母以上）请用缩写代替
（4）所有参数需要有默认值
（5）参数个数限制在10个之下

 */
package com.shu;
import Jama.Matrix;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
public class Test{
    public static void main(String [ ] args) throws Exception{
        int lines=0;
        int features=0;
        String temp;
        String[] ss;
        BufferedReader br1 = new BufferedReader(new FileReader("C:\\Users\\ggxxding\\IdeaProjects\\PCA_Softmax\\wine.data.txt"));//
        while((temp=br1.readLine())!=null){
            ss=temp.trim().split("\\,+");
            if(temp.isEmpty()){
                break;
            }
            lines++;
            features=ss.length-1;////
        }
        int[] tag=new int[lines];//
        double[][] x=new double[features][lines];
        BufferedReader br = new BufferedReader(new FileReader("C:\\Users\\ggxxding\\IdeaProjects\\PCA_Softmax\\wine.data.txt"));
        int i=0;
        while((temp=br.readLine())!=null){

            ss=temp.trim().split("\\,+");
            if(temp.isEmpty()){
                break;
            }
            tag[i]=Integer.parseInt(ss[0]);
            for(int j=0;j<features;j++){
                x[j][i]=Double.parseDouble(ss[j+1].toString());
            }
            i++;
        }
        Matrix X=new Matrix(x);

        //X.print(4,2); //X为列向量
        PCA pca=new PCA(X);
        pca.getPercent();
        SoftmaxRegression SRtest=new SoftmaxRegression(pca.dimReduct(5),tag,3);
        SRtest.fit(1000,0.05,0.01);
        SRtest.predict();
        SRtest.score();
        System.exit(0);
    }
    /*读取字符串分割转换为数字，参考
    static int[] aryChange(String temp) {// 字符串数组解析成int数组
        String[] ss = temp.trim().split("\\,+");// .trim()可以去掉首尾多余的空格
        // .split("\\s+")
        // 表示用正则表达式去匹配切割,\\s+表示匹配一个或者以上的空白符
        int[] ary = new int[ss.length];
        for (int i = 0; i < ary.length; i++) {
            ary[i] = Integer.parseInt(ss[i]);// 解析数组的每一个元素
        }
        return ary;// 返回一个int数组
    }*/
    /*Jama.Matrix包例子
    import Jama.Matrix;  // 导入Jama包中的Matrix类

public class helloworld {

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        System.out.println("调用jama包完成矩阵运算");

        double [][] array = {
                {-1, 1, 0},
                {-4, 3, 0},
                {1, 0, 2}
        };

        System.out.println("特征分解");
        // http://zhidao.baidu.com/link?url=KZY21A85_YfXLCw-4dZlES5AdgjXkQg4uuogjLsv6WvGV3VM9sBkTOQUofPpEzRqSO0WwlVrBMi8e-9hd4Rhoa
        Matrix A = new Matrix(array);
        A.eig().getD().print(4, 2);   // 对角元素是特征值，4是列的宽度，2代表小数点后的位数
        A.eig().getV().print(4, 2);   // 特征向量

        System.out.println("矩阵维数");
        int rowNum = A.getRowDimension();  // 矩阵行数
        int colNum = A.getColumnDimension();
        System.out.println(rowNum + " " + colNum);

        System.out.println("行列式");
        double detNum = A.det();   // 行列式
        System.out.println(detNum);
    }

}
    */
}
