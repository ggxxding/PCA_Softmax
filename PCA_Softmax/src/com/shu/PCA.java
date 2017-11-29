package com.shu;
import Jama.Matrix;
import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
/*
编译单元在这个情况下是一个源码文件，应被赋予文件内定义的主要的类的名字。
用与类相同的名字命名文件，扩展名.java作为后缀名。示例：RandomForest.java。
成员函数的命名应采用完整的英文描述符，大小写混合使用：所有中间单词的首字母大写.
有四个函数在类中一定要有，
分别是：
用于初始化的构造函数（构造函数名与类名相同），
训练模型函数fit，
预测函数predict，
评分函数score。
 */
public class PCA {
    Matrix X;               //X为样本列向量构成矩阵
    Matrix matrix_cov;       //协方差矩阵
    Matrix eig_values;        //特征值对角矩阵
    Matrix eig_vectors;       //特征向量矩阵
    int column;             //样本向量个数
    int row;                //样本维数
    int[] eig_index;      //特征值从大到小的序列

    public PCA(Matrix x){   //初始化,计算特征值和特征向量等
        double temp;        //临时变量
        X=x;
        column=X.getColumnDimension();
        row=X.getRowDimension();
        double[] xMean=new double[row];//样本均值
        double[] xVar=new double[row];//样本方差
        for(int i=0;i<row;i++){
            xMean[i]=0;
        }
        for(int i=0;i<row;i++){         //求样本均值
            for(int j=0;j<column;j++){
                xMean[i]+=X.get(i,j);
            }
            xMean[i]=xMean[i]/column;
        }
        for(int i=0;i<row;i++){         //样本中心化
            for(int j=0;j<column;j++){
                temp=X.get(i,j)-xMean[i];
                X.set(i,j,temp);
            }
        }
        for(int i=0;i<row;i++){
            xVar[i]=0;
        }
        for(int i=0;i<row;i++){         //求样本方差
            for(int j=0;j<column;j++){
                xVar[i]=xVar[i]+Math.pow(X.get(i,j),2);
            }
            xVar[i]=xVar[i]/column;
        }
        for(int i=0;i<row;i++){         //样本方差归一化
            for(int j=0;j<column;j++){
                temp=X.get(i,j)/Math.pow(xVar[i],0.5);
                X.set(i,j,temp);
            }
        }
        matrix_cov=X.times(X.transpose());  //计算协方差矩阵
        matrix_cov=matrix_cov.times(1.0/column);
        eig_values=matrix_cov.eig().getD();               //特征值
        eig_vectors=matrix_cov.eig().getV();              //特征向量
        double[] eigSort=new double[row];               //排序的特征值
        for(int i=0;i<row;i++){
            eigSort[i]=eig_values.get(i,i);
        }
        Arrays.sort(eigSort);                           //从小到大排序
        eig_index=new int[row];
        int[] flag=new int[row];
        for(int i=0;i<row;i++){
            flag[i]=0;
        }
        for(int i=row-1;i>=0;i--){                      // 生成eigIndex
            for(int j=0;j<row;j++){
                if(eigSort[i]==eig_values.get(j,j)){
                    if(flag[j]==0){
                        flag[j]=1;
                        eig_index[row-i-1]=j;
                        break;
                    }
                }
            }
        }
    }
    public void getPercent(){//获得各个特征值的大小比例
        double[] temp=new double[row];
        double sum=0;
        for(int i=0;i<row;i++){
            temp[i]=eig_values.get(i,i);
            sum+=temp[i];
        }

        for(int i=0;i<row;i++){
            System.out.println(eig_index[i]+" the "+ (i+1)+"th eigenValue is: "+temp[i]+" Percentage: "+ temp[i]/sum*100 +"%");
        }////////////////////////////////////
    }
    public Matrix dimReduct(int dim){     //降维，输入希望降到的维度，返回降维后的列向量矩阵
        double[][] temp=new double[row][dim];
        for(int n=0;n<dim;n++){
            for(int i=0;i<row;i++){
                temp[i][n]=eig_vectors.get(i,eig_index[n]);
            }
        }
        Matrix W=new Matrix(temp);
        return W.transpose().times(X);
    }
    public Matrix getW(int dim){            //根据希望得到的维度返回投影矩阵
        double[][] temp=new double[row][dim];
        for(int n=0;n<dim;n++){
            for(int i=0;i<row;i++){
                temp[i][n]=eig_vectors.get(i,eig_index[n]);
            }
        }
        Matrix W=new Matrix(temp);

        return W;
    }

}
