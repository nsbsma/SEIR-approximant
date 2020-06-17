function [A,Sinf,a0,a]=ApproximantCoefficientsSIR(N,alpha,r,S0,I0)
% This computes the A_n coefficients and Sinf needed for the N-term SIR approximant
% given as equation (12a) in Barlow & Weinstein, Physica D 408, 1 (2020) (https://doi.org/10.1016/j.physd.2020.132540)
% The inputs correspond to the SIR parameters and initial conditions as
% specified by equation (1). Python code available at https://github.com/nsbsma/SIR-approximant
% 
% 
% %%% example, reproducing figure 2b in Barlow & Weinstein, Physica D 408, 1 (2020)
% alpha=2.73; r=0.0178; I0=7; S0=254; %input parameters
% t=0:0.01:6 %time interval 0 to 6 in increments of 0.01
% N=25; %number of terms in the approximant, increase until answer stops changing
% [A,Sinf,a0,a]=ApproximantCoefficientsSIR(N,alpha,r,S0,I0); %using the code provided to get the stuff needed below
% kappa=r*Sinf-alpha;
% Ssum=1;
%     for j=1:N
%         Ssum=Ssum+A(j)*exp(kappa*t*j); %approximant (equation 4.7a in paper)
%     end
% S=Sinf./Ssum;
% plot(t,S,'displayname','S'); hold on
% I=alpha/r*log(S/S0)+S0-S+I0;
% plot(t,I,'displayname','I');
% R=S0+I0-S-I; %b/c S+I+R=constant
% plot(t,R,'displayname','R'); xlabel('t'); legend show

beta=alpha*log(S0)-r*(S0+I0);
a0=S0; b0=log(a0);at0=1/a0; %zero index of things, since MATLAB starts at 1
a=zeros(1,N); at=a; b=a; %don't need this, but speeds up MATLAB, preallocates memory
a(1)=beta*a0+a0*(r*a0-alpha*b0);
for n=1:N-1
    
    %%equation 6d
    sum1=a(n)*at0;
    for j=1:n-1
        sum1=sum1+a(j)*at(n-j);
    end
    at(n)=-sum1/a0;
    
    %%equation 6c
    sum2=n*a(n)*at0;
    for j=0:n-2
        sum2=sum2+(j+1)*a(j+1)*at(n-1-j);
    end
    b(n)=sum2/n;
    
    %%equation 6b
    asum=a0*(r*a(n)-alpha*b(n))+a(n)*(r*a0-alpha*b0);
    for j=1:n-1
        asum=asum+a(j)*(r*a(n-j)-alpha*b(n-j));
    end
    a(n+1)=(asum+beta*a(n))/(n+1);
end

 %iterative solver for equation 7
syms Sinf
f=alpha/r*log(Sinf/S0)+(S0-Sinf)+I0;
digits(32);
Sinf=double(vpasolve(f,Sinf,0.1));

%%equation 12 stuff/formation
kappa=r*Sinf-alpha; 
aa0=at0*Sinf; aa=at*Sinf;
% M(1,:)=ones(N,1); %only need if not using VDMinv function
b(1)=aa0-1; %first entry of equation 12c
for j=1:N-1
%     M(j+1,:)=(1:N).^j; %only need if not using VDMinv function
    b(j+1)=aa(j)*factorial(j)*(1/kappa)^(j); %equation 12c
end
Minv=VDMinv(N); %implementing fast VanderMonde inverter
% Minv=inv(M); %only need if not using VDMinv function
% A=b.'\M; %only need if not using VDMinv function
A=Minv*b.'; %Minv times the transpose of b


function f=VDMinv(N) %Vandermonde matrix inverter. 
L=zeros(N);U=L;
L(1,1)=1;
for i=1:N
    for j=1:N
        if j*i~=1 && i>=j
            prod=1;
            for k=1:i
                if j~=k
                    prod=prod/(j-k);
                else
                end
            end
            
            L(i,j)=prod;
        else
        end
        if j==i
            U(i,j)=1;
        elseif j~=1
            if i==1
                U(i,j)=-U(i,j-1)*(j-1);
            else
            U(i,j)=U(i-1,j-1)-U(i,j-1)*(j-1);
            end
        end
    end
end
f=(U*L).';