
## Problem

$$ min\ f = x_1^2 +(x_2-3)^2 $$

S.T.

$$ g_1: x_2^2-2x_1 \leq 0 $$

$$ g_2: (x_2-1)^2+5x_1-15 \leq 0 $$

## QP subproblem

$$ L =  x_1^2 +(x_2-3)^2 + \mu_1( x_2^2-2x_1) + \mu_2((x_2-1)^2+5x_1-15)$$

$$ \nabla L = \begin{pmatrix}2x_1-2\mu _1+5\mu _2\\
2x_2-6+2\mu _1\cdot x_2+\mu _2(2x_2-2)\end{pmatrix}$$

## Code
```
f= @(x) x(1)^2 +(x(2)-3)^2;
df = @(x) [2*x(1) , 2*(x(2)-3)];
g = @(x) [x(2)^2-2*x(1) ; (x(2)-1)^2+5*x(1)-15];
dg = @(x) [-2 2*x(2) ; 5 2*(x(2)-1)];

X = [1; 1]; %initial conditions, make sure its feasible
e = .01;
mu_prev = [0; 0]; %initial mu's
w_prev= [0 , 0]; %initial weights for line search

magGradLagr = norm(df(X)+ mu_prev'*dg(X)) %magnitute of gradient of lagrangian

W = [1, 0; 0, 1];
while magGradLagr>e

quadprogParameters = optimset('Algorithm', 'active-set', 'Display', 'off');
 [s,~,~,~,lambda] = quadprog(W,[df(X)]',dg(X),-g(X),[], [], [], [], [1,1],  quadprogParameters); %solve qp problem
            mu_next = lambda.ineqlin;

%a= .1; %no linesearch

%linesearch

b = .8 ;
t = .1;
a =1; %initial a value
N=100;

w_next = max(abs(mu_prev), .5*(w_prev+abs(mu_prev)));

i=0;
  while i<N
    %calc phi
        phi_a = f(X + a*s) + w_next'*abs(min(0, -g(X+a*s)));
        
        phi_0 = f(X) + w_next'*abs(min(0, -g(X)));        
        dphi_0 = df(X)*s + w_next'*((dg(X)*s).*(g(X)>0));
        psi_a = phi_0 +  t*a*dphi_0;                

        if phi_a<psi_a
            break;
        else
         
            a = a*b;
            i = i + 1;
        end
  end

dx = a*s;
X=X+dx; %gradient decent

%update hessian

 y = [df(X) + mu_next'*dg(X) - df(X-dx) - mu_next'*dg(X-dx)]'; 
       
        if dx'*y >= 0.2*dx'*W*dx
            theta = 1;
        else
            theta = (0.8*dx'*W*dx)/(dx'*W*dx-dx'*y);
        end
        
        dg_k = theta*y + (1-theta)*W*dx;

        W = W + (dg_k*dg_k')/(dg_k'*dx) - ((W*dx)*(W*dx)')/(dx'*W*dx); %rank 2 update
        
   
        magGradLagr = norm(df(X) + mu_next'*dg(X)); % norm of Largangian gradient
        mu_prev = mu_next;



end
X
```

## Final Results

X1 = 1.0602
X2 = 1.4562
