%%
clear
load('./cir_coords.mat')
%load('/home/yiming/yiming_research/invariant_neural_operators-main/data_linear_peridynamic_solid/cir_coords05.mat')
domain1_coords=cir_coords(1:53,:);
%[xx,yy]=meshgrid(-0.4:0.1:0.4,-0.4:0.1:0.4);
[xx,yy]=meshgrid(-0.4:0.05:0.4,-0.4:0.05:0.4);
xx=xx(:);
yy=yy(:);
rr=xx.^2+yy.^2;
kk=floor(0.8/0.05)+1;
for i=1:kk^2
    if rr(i)<0.4^2
        domain1_coords=[domain1_coords;[xx(i),yy(i)]];
    end
end
figure(1)
scatter(domain1_coords(:,1),domain1_coords(:,2))
hold on
rectangle('Position', [[-0.4 -0.4], 2*0.4, 2*0.4], 'Curvature', [1,1], 'EdgeColor', 'b');
xlabel('x')
ylabel('y')
%set(gca,'FontSize',20)
pause(1)

Nh=41;
dh=1/(Nh-1);
[X,Y]=meshgrid(-0.5:dh:0.5, -0.5:dh:0.5);
ul_all=0;
ucla=0;
j=0;
for i=201:300
    dir='ubctest_final_order2_0075_0075_coef5';
    uname=['/home/yiming/yiming_research/fenics/HGO2d/',dir,'/displacement_',num2str(i-1),'.txt'];
    fname=['/home/yiming/yiming_research/fenics/HGO2d/',dir,'/bodyforce_',num2str(i-1),'.txt'];
    
    %filename = 'your_file.txt';  % Specify the file name or path
    if exist(uname, 'file') == 2
        j=j+1;
        u_all=load(uname);
        f_all=load(fname);
        %{
        figure(2)
        subplot(2,2,1)
        surf(reshape(u_all(:,1),Nh,Nh))
        title('ux')
        view(2)
        grid off
        colorbar
        xlim([1 41])
        ylim([1 41])

        subplot(2,2,2)
        surf(reshape(u_all(:,2),Nh,Nh))
        title('uy')
        view(2)
        grid off
        colorbar
        xlim([1 41])
        ylim([1 41])

        subplot(2,2,3)
        surf(reshape(f_all(:,1),Nh,Nh))
        title('fx')
        view(2)
        grid off
        colorbar
        xlim([1 41])
        ylim([1 41])

        subplot(2,2,4)
        surf(reshape(f_all(:,2),Nh,Nh))
        title('fy')
        view(2)
        grid off
        colorbar
        xlim([1 41])
        ylim([1 41])

        %saveas(gcf, ['/home/yiming/yiming_research/fenics/HGO2d/',dir,'/uf',num2str(i-1),'.png']);  % Save as PNG file
        %pause(1)
        %}

        newu1=interp2(X,Y,reshape(u_all(:,1),Nh,Nh),domain1_coords(:,1),domain1_coords(:,2),'spline');
        newu2=interp2(X,Y,reshape(u_all(:,2),Nh,Nh),domain1_coords(:,1),domain1_coords(:,2),'spline');

        newf1=interp2(X,Y,reshape(f_all(:,1),Nh,Nh),domain1_coords(:,1),domain1_coords(:,2),'spline');
        newf2=interp2(X,Y,reshape(f_all(:,2),Nh,Nh),domain1_coords(:,1),domain1_coords(:,2),'spline');

        u(j,:)=[newu1',newu2'];
        f(j,:)=[newf1',newf2'];

        %simple diff test
        ux1=reshape(u_all(:,1),Nh,Nh);
        uy1=reshape(u_all(:,2),Nh,Nh);
        n=Nh;
        dux1=(ux1(:,3:n)-ux1(:,1:n-2))/(1/(n-1)*2);
        duy1=(uy1(3:n,:)-uy1(1:n-2,:))/(1/(n-1)*2);
        div=dux1(2:n-1,:)+duy1(:,2:n-1);
        dn=4;
        divl=sqrt(sum(sum(div(dn:n-dn-1,dn:n-dn-1).^2))/((n-2*dn-1)^2));
        ul=sqrt(sum(sum(ux1(dn+1:n-dn,dn+1:n-dn).^2)+sum(uy1(dn+1:n-dn,dn+1:n-dn).^2))/((n-2*dn-1)^2));
        
        fprintf('i=%d, j=%d, ',i, j)
        
        fprintf('simple div test: %f, ', divl/ul)
        ul_all=ul_all+ul;
        
        ucl=norm(u(j,:));
        ucla=ucla+ucl;
        %ccl=3.424909; 
        ccl=norm(domain1_coords);
        fprintf('u norm: %f, coord norm: %f\n', ucl, ccl)
        
        if mod(j,100)<100
        figure(3)
        subplot(2,3,1)
        surfir(domain1_coords(:,1),domain1_coords(:,2),newu1);
        title('ux')
        view(2)
        grid off
        colorbar

        subplot(2,3,4)
        surfir(domain1_coords(:,1),domain1_coords(:,2),newu2);
        title('uy')
        view(2)
        grid off
        colorbar

        subplot(2,3,2)
        surfir(domain1_coords(:,1),domain1_coords(:,2),newu1+domain1_coords(:,1));
        title('ux+x')
        view(2)
        grid off
        colorbar

        subplot(2,3,5)
        surfir(domain1_coords(:,1),domain1_coords(:,2),newu2+domain1_coords(:,2));
        title('uy+y')
        view(2)
        grid off
        colorbar

        subplot(2,3,3)
        surfir(domain1_coords(:,1),domain1_coords(:,2),newf1);
        title('fx')
        view(2)
        grid off
        colorbar

        subplot(2,3,6)
        surfir(domain1_coords(:,1),domain1_coords(:,2),newf2);
        title('fy')
        view(2)
        grid off
        colorbar
        
        fig=gcf;
        fig.Position(3:4) = [1400, 600];
        %pause(2)

        %saveas(gcf, ['/home/yiming/yiming_research/fenics/HGO2d/',dir,'/uf',num2str(i-1),'_j',num2str(j),'.png']);  % Save as PNG file
        end
        
    
    end
    
end
fprintf('u norm avg: %f, coord norm: %f, ratio: %f\n',ucla/j, ccl, ucla/j/ccl)
coords=domain1_coords;
save(['./',dir,'_domain0.mat'],'u','f','coords')
