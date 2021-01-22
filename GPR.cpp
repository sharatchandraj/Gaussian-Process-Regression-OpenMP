#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>

using namespace std;
using namespace std::chrono;

int GRID_SIZE;
float rstar_x;
float rstar_y;


template<class T>
void print_matrix(T **A, int r,int c){
    for(int i= 0; i<r;i++){
        for(int j=0;j<c;j++){
            cout<<A[i][j]<<"\t"<<flush;
        }
        cout<<endl;
    }
    cout<<endl;
}

//function to generate matrix to hold the points' coordinates
float **generate_grid_points(){
//    srand(time(NULL));
    int n= GRID_SIZE*GRID_SIZE;
    float h=1.0/(GRID_SIZE+1);
    int idx=0;

    float **A = new float*[n];
    for(int i = 0; i <n; i++)
        A[i] = new float[2];

    for(int i= 0; i<n;i++){
        for(int j=0;j<2;j++){
            A[i][j]=0;
        }
    }


    for (int i = 1; i <= GRID_SIZE; i++) {
        for (int j = 1; j <=GRID_SIZE; j++) {
            A[idx][0]=i*h;
            A[idx][1]=j*h;
            idx+=1;
        }

    }

    return A;
}

float *generate_observed_data(float **A){

    float temp = 0.0;
    int n=GRID_SIZE*GRID_SIZE;
    float *f = new float[n];
    int i;

    cout<<"f Matrix!!"<<endl;

    for (i=0; i<n; i++) {
        f[i] = ((float) 0.01 * ((rand() % 11) - 5));
        f[i] = f[i]+ 1.0 - ( pow((A[i][0] - 0.5),2.0) + pow((A[i][1] - 0.5),2.0));

        cout<<f[i]<<endl;
    }

    cout<<endl;

    return f;

}


float **compute_K(float **A){

    int n=GRID_SIZE*GRID_SIZE;
    int i,j;
    float temp,d;

    float **K = new float*[n];
    for( i=0;i<n;i++){
        K[i] = new float[n];
    }


    #pragma omp barrier
    #pragma omp parallel for shared(K) private(i,j,temp)
    for( i=0;i<n;i++){
        for( j=0;j<n;j++){
           if(i==j)
                K[i][j]=1.0;
           else{
                temp = pow((A[i][0] - A[j][0]),2.0) + pow((A[i][1] - A[j][1]),2.0);
                K[i][j] = exp(-1.0 * temp );

           }
        //   cout<<K[i][j]<<"\t ";

        }
        // cout<<endl;
    }

    return K;
}

float *generate_k(float **A,float rstar_x,float rstar_y){

    int n=GRID_SIZE*GRID_SIZE;
    int i;

    float *k = new float[n];

    cout<<"k Matrix!!!"<<endl;

    #pragma omp barrier
    #pragma omp parallel for shared(A, rstar_x, rstar_y, k) private(i)
    for(i=0;i<n;i++){
        k[i] = pow((A[i][0] - rstar_x),2.0) + pow((A[i][1] - rstar_y),2.0);
        k[i] = exp(-1.0 * k[i]);
        cout<<k[i]<<endl;
    }

    cout<<endl;

    return k;
}

void LU(float **A){

    int n=GRID_SIZE*GRID_SIZE;
    int i,j,k;
    float m = 0;


    for( i=0;i<n-1;i++){
        #pragma omp barrier
        #pragma omp parallel for private(j,k,m) shared(A)
        for( j=i+1;j<n;j++){
            m= A[j][i]/A[i][i];
            for( k=i+1;k<n;k++){
                A[j][k]=A[j][k]-m*A[i][k];
            }
            A[j][i]=m;
        }
    }

}

void extract_LU(float**L, float**U, float **K){

    int n=GRID_SIZE*GRID_SIZE;
    int i,j;

    //extract U
    #pragma omp parallel for private(i,j) shared(L,U,K)
    for(i=0; i<n; i++){
        for(j=0; j<n; j++){
            if(i<j){
                L[i][j] = 0;
                U[i][j] = K[i][j];
            }
            else if (i==j){
                L[i][j] = 1.0;
                U[i][j] = K[i][j];
            }
            else {
                L[i][j] = K[i][j];
                U[i][j] = 0;
            }

            // cout<<L[i][j]<<"\t";
        }
        // cout<<endl;
    }
}

void compute_fstar(float **L, float **U, float *k, float *f){

    int n=GRID_SIZE*GRID_SIZE;
    float fstar=0;
    float *z=new float[n];
    float *y=new float[n];
    float temp=0.0;
    int i,j;

    /////////forward substitution//////////////////
    for(i=0;i<n;i++){
        temp=0;
        //#pragma omp parallel for reduction(+:temp)
        for(j=0;j<i;j++){
            temp=temp+L[i][j]*y[j];
        }
        y[i] = (f[i] - temp)/L[i][i];
    }

    cout<<"z Matrix"<<endl;
    ////////backward substitution///////////////
    for(i=n-1;i>-1;i--){
        temp = 0.0;
        //#pragma omp parallel for reduction(+:temp)
        for(j=n-1;j>i;j--){
         temp = temp + U[i][j]*z[j];
        }
        z[i]= (y[i]-temp)/U[i][i];
        cout<<z[i]<<endl;
    }

    cout<<endl;


    #pragma omp parallel for reduction(+:fstar)
    for(i=0;i<n;i++){
        fstar=fstar+k[i]*z[i];
    }


    cout<<"Predicted value of function at rstar: "<<fstar<<endl;

    delete [] y;
    delete [] z;



}

int main(int argc, char* argv[]){

    srand(time(0));

    if(argc>1){
        GRID_SIZE= atoi( argv[1] );
        rstar_x=atof( argv[2] );
        rstar_y=atof( argv[3] );

        cout<<GRID_SIZE<<", "<<rstar_x<<", "<<rstar_y<<endl;
    }else{
        GRID_SIZE= 2;
        rstar_x=0.5;
        rstar_y=0.5;

        cout<<GRID_SIZE<<", "<<rstar_x<<", "<<rstar_y<<endl;
    }

    int n=GRID_SIZE*GRID_SIZE;

    float **X= generate_grid_points();
    // print_matrix(X,n,2);

    float *f = generate_observed_data(X);


    float **K = compute_K(X);


    float *k = generate_k(X,rstar_x,rstar_y);

//    cout<<"k Matrix!!!!"<<endl;
//    print_matrix(k , n , 1);

    float **L = new float*[n];
    float **U = new float*[n];
    for(int i=0;i<n;i++){
        L[i] = new float[n];
        U[i] = new float[n];
    }

    //computing (tI+K) and storing it in K
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            if(i==j){
                K[i][j]=0.01+K[i][j];
            }
        }
    }

    cout<<"K Matrix!!!!"<<endl;
    print_matrix(K , n , n);

    auto start = std::chrono::steady_clock::now( );

    //LU factorization
    LU(K);

    //extract L and U matrices
    extract_LU(L,U,K);

    //compute fstar
    compute_fstar(L,U,k,f);



    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>( std::chrono::steady_clock::now( ) - start );

    cout<<"Time taken to compute fstar: "<<elapsed.count()<<"ms"<<endl;

    delete [] k;
    delete [] f;
    for(int i=0;i<n;i++){
        delete [] X[i];
        delete [] K[i];
        delete [] U[i];
        delete [] L[i];
    }

    delete [] U;
    delete [] L;
    delete [] X;
    delete [] K;
return 0;
}
