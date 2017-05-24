M=10; L=5;
nSym=10;
nMC=100;
nSNRpoints=50;

SNR=zeros(1,nSNRpoints);

BER1=zeros(1,nSNRpoints);
BER2=zeros(1,nSNRpoints);
BER3=zeros(1,nSNRpoints);
BER4=zeros(1,nSNRpoints);

MSE1=zeros(1,nSNRpoints);
MSE2=zeros(1,nSNRpoints);
MSE3=zeros(1,nSNRpoints);
MSE4=zeros(1,nSNRpoints);

for v=1:nSNRpoints
    b_cnt1=0;
    b_cnt2=0;
    b_cnt3=0;
    b_cnt4=0;
    mse1=0;
    mse2=0;
    mse3=0;
    mse4=0;
  
    for n=1:nMC
        a=randi(4,M,nSym)-1;              
        s=pskmod(a,4);
        s_mag=M*nSym;
        
        %for ZP
        F=eye(M);           %F can be changed
        ZP1=[F;zeros(L,M)];
        
        %channel and DH_inv
        h_rand=1/sqrt(2) * (randn(L+1,1) + 1i * randn(L+1,1));   
        h=reshape(h_rand,1,L+1);
        H = fft([h(:); zeros(M-L-1, 1)]);
        DH_inv=inv(diag(H));
        
        %noise
        w=1/sqrt(2)*(randn(M+L,nSym)+ 1i *randn(M+L,nSym));
        w_mag=0;
        for p=1:(M+L)
            for q=1:nSym
                w_mag=w_mag+(abs(w(p,q)))^2;    %square
            end
        end
        SNR(v)=0.1+(999.9/nSNRpoints)*v;                     %SNR=square/square    %SNR range:0.1~1000
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %CP
        S=ifft(s)*sqrt(M);               
        CP1=[zeros(L,M-L),eye(L);eye(M)];
        U=CP1*S;                         
        u=reshape(U,1,nSym*(M+L));
        R=filter(h,1,u);
        r=reshape(R,M+L,nSym);
        R1_mag=0;        
        for p=1:(M+L)
            for q=1:nSym
                R1_mag=R1_mag+(abs(r(p,q)))^2;    %square
            end
        end
        w1=w*sqrt(R1_mag/(w_mag*SNR(v)));   
        r=r+w1;
        CP2=[zeros(M,L),eye(M)];         
        r1_2=CP2*r;                       
        r1_3=fft(r1_2)/sqrt(M);
        
        G=zeros(M,1);
        for i=1:M
            G(i)=conj(H(i))*s_mag/((abs(H(i)))^2*s_mag+R1_mag/SNR(v));
        end
        DG=diag(G);
        
        s_hat1=DH_inv*r1_3;
        s_hat4=DG*r1_3;
        a_hat1=pskdemod(s_hat1,4);
        a_hat4=pskdemod(s_hat4,4);
        b_cnt1=b_cnt1+biterr(a,a_hat1);
        b_cnt4=b_cnt4+biterr(a,a_hat4);
        Error1 = s_hat1 - s;
        mse1 = mse1+ mean(mean(abs(Error1).*abs(Error1)));
        Error4 = s_hat4 - s;
        mse4 = mse4+ mean(mean(abs(Error4).*abs(Error4)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %ZP1
        U2=ZP1*s;
        u2=U2(1:M,:);
        h2_conv=toeplitz([h,zeros(1,M+L-length(h))],[h(1),zeros(1,M-1)]);
        u2_conv=h2_conv*u2;
        R2_mag=0;
        for p=1:(M+L)
            for q=1:nSym
                R2_mag=R2_mag+(abs(u2_conv(p,q)))^2;    %square
            end
        end
        w2=w*sqrt(R2_mag/(w_mag*SNR(v)));
        u2_conv=u2_conv+w2;
        h2_pi=inv(h2_conv'*h2_conv)*h2_conv';
        r2=h2_pi*u2_conv;
        s_hat2=inv(F)*r2;
        a_hat2=pskdemod(s_hat2,4);
        b_cnt2=b_cnt2+biterr(a,a_hat2);
        Error2 = s_hat2 - s;
        mse2 = mse2+ mean(mean(abs(Error2) .* abs(Error2)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        %ZP2
        U3=ZP1*s;
        u3=reshape(U3,1,(M+L)*nSym);
        R3_1=filter(h,1,u3);
        r3_1=reshape(R3_1,(M+L),nSym);
        R3_mag=0;
        for p=1:(M+L)
            for q=1:nSym
                R3_mag=R3_mag+(abs(r3_1(p,q)))^2;    %square
            end
        end
        w3=w*sqrt(R3_mag/(w_mag*SNR(v)));
        r3_1=r3_1+w3;
        ZP2=[eye(L),zeros(L,M-L),eye(L);zeros(M-L,L),eye(M-L),zeros(M-L,L)];
        r3_2=ZP2*r3_1;
        R3_2=fft(r3_2)/sqrt(M);
        R3_3=DH_inv*R3_2;
        r3_3=ifft(R3_3)*sqrt(M);
        s_hat3=inv(F)*r3_3;
        a_hat3=pskdemod(s_hat3,4);
        b_cnt3=b_cnt3+biterr(a,a_hat3);
        Error3 = s_hat3 - s;
        mse3 = mse3+ mean(mean(abs(Error3) .* abs(Error3)));
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    BER1(v)=b_cnt1/nMC/M/nSym;
    BER2(v)=b_cnt2/nMC/M/nSym;
    BER3(v)=b_cnt3/nMC/M/nSym;
    BER4(v)=b_cnt4/nMC/M/nSym;
    MSE1(v)=mse1/nMC;
    MSE2(v)=mse2/nMC;
    MSE3(v)=mse3/nMC;
    MSE4(v)=mse4/nMC;
end

figure(1)
semilogy(log10(SNR),BER1);
hold on
semilogy(log10(SNR),BER2);
hold on
semilogy(log10(SNR),BER3);
hold on
semilogy(log10(SNR),BER4);
xlabel('SNR')
ylabel('BER')
title(['M = ', num2str(M), '; L = ', num2str(L), '; nSym = ', num2str(nSym), '; Random Rayleigh channel; nMC = ', num2str(nMC)])
legend('CP','ZP1','ZP2','CP:MMSE')

figure(2)
semilogy(log10(SNR),MSE1);
hold on
semilogy(log10(SNR),MSE2);
hold on
semilogy(log10(SNR),MSE3);
hold on
semilogy(log10(SNR),MSE4);
xlabel('SNR')
ylabel('MSE')
title(['M = ', num2str(M), '; L = ', num2str(L), '; nSym = ', num2str(nSym), '; Random Rayleigh channel; nMC = ', num2str(nMC)])
legend('CP','ZP1','ZP2','CP:MMSE')
