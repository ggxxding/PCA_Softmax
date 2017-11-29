package com.shu;
        import Jama.Matrix;
        import java.io.*;
        import java.lang.reflect.Parameter;
        import java.util.ArrayList;
        import java.util.Arrays;
        import java.util.HashSet;
        import java.util.List;
        import java.util.Set;

        import static java.lang.Math.exp;
public class SoftmaxRegression {
    Matrix X;               //X为样本列向量构成矩阵
    int column;             //样本向量个数
    int row;                //样本属性个数
    int[] tags;             //样本标签
    int num_tags=0;        //类别数
    int[] predict_tags;      //预测结果
    Matrix parameterW;               //参数
    public SoftmaxRegression(Matrix x,int[] tag,int numOfTag){

        X=x;//传入样本数据
        column=X.getColumnDimension();
        row=X.getRowDimension();
       tags=tag;
       num_tags=numOfTag;
       predict_tags=new int[column];
    }

    public void fit(int steps,double alpha,double lambda){//训练 steps迭代步数 alpha学习率 lambda正则化权重 parameterW是学习结果参数
        double[][] w=new double[num_tags][row];
        double[][] wGradient=new double[num_tags][row];
        Matrix temp=new Matrix(new double[1][row]);
        for(int i=0;i<num_tags;i++){
            for(int j=0;j<row;j++){
                w[i][j]=1.0;
                wGradient[i][j]=0.0;
            }
        }
        Matrix W=new Matrix(w);                 //存储参数矩阵 行向量存储
        Matrix WGradient=new Matrix(wGradient); //存储梯度矩阵 行向量存储
        //compute gradient
        int flagI;  //对应I{y=j}
        double p;
        double pTemp;
        int step=0;
        while(step<steps) {
            for (int i = 0; i < num_tags; i++) {//计算第i个参数向量的梯度
                for (int t = 0; t < row; t++) {
                    temp.set(0, t, 0);
                }

                for (int j = 0; j < column; j++) {//计算Wi梯度
                    pTemp = 0;
                    if (tags[j]==i+1) {
                        flagI = 1;
                    } else {
                        flagI = 0;
                    }
                    for (int k = 0; k < num_tags; k++) {
                        pTemp += exp(W.getMatrix(k, k, 0, row - 1).times(X.getMatrix(0, row - 1, j, j)).get(0, 0));
                    }
                    p = exp(W.getMatrix(i, i, 0, row - 1).times(X.getMatrix(0, row - 1, j, j)).get(0, 0)) / pTemp;
                    temp = temp.plus(X.getMatrix(0, row - 1, j, j).times(flagI - p).transpose());
                }
                temp = temp.times(-1.0 / column).plus(W.getMatrix(i,i,0,row-1).times(lambda));//temp是梯度
                for (int m = 0; m < row; m++){
                    WGradient.set(i, m, temp.get(0, m));
                }
            }
            W=W.minus(WGradient.times(alpha));
            step++;
        }
        parameterW=W;
    }
    public void predict(){//预测,将结果存入类中的predictTags[]
        double temp;
        double max;
        double[] p=new double[num_tags];
        int flag;
        for(int i=0;i<column;i++){//预测第i个样本
            temp=0;
            for(int j=0;j<num_tags;j++){//预测第i个样本属于第j类的概率
                temp=0;
                for(int k=0;k<num_tags;k++){//概率分母求和
                    temp+=exp(parameterW.getMatrix(k,k,0,row-1).times(X.getMatrix(0,row-1,i,i)).get(0,0));
                }
                temp=exp(parameterW.getMatrix(j,j,0,row-1).times(X.getMatrix(0,row-1,i,i)).get(0,0))/temp;
                p[j]=temp;
            }
            flag=0;
            for(int l=1;l<num_tags;l++){//判定归类
                if(p[flag]<p[l]){
                    flag=l;
                }
            }
            predict_tags[i]=flag+1;
            System.out.println(predict_tags[i]);
        }
    }
    public void score(){//评分
        double num_wrong;
        double rate;
        num_wrong=0;
        for(int i=0;i<column;i++){
            if(predict_tags[i]!=tags[i]){
                num_wrong=num_wrong+1;
            }
        }
        rate=num_wrong/column*100.0;
        System.out .println("wrong rate:"+rate+"%");
    }
}
