function discreteregularization
% This function solves the multi-bang regularized inverse source problem
%  min 1/2 \|y-yd\|^2 + alpha G(u)
%      s.t. -\Delta y = u,  u_1 <= u(x) <= u_d
% using the approach described in the paper
%    "Convex regularization of discrete-valued inverse problems"
% by Christian Clason and Thi Bich Tram Do, see
% http://arxiv.org/abs/1707.01041.
%
% July 4, 2017               Christian Clason <christian.clason@uni-due.de>
%                                     Thi Bich Tram Do <tram.do@uni-due.de>

%% setup
% problem parameters
N     = 128;                            % number of nodes per dimension
maxit = 100;                            % max number of Newton steps
alpha = 1e-4;                           % regularization parameter

ub = [0 0.1 0.15]';                     % vector of parameter values
d  = length(ub);                        % number of parameter values
delta = 0.5;

% setup grid, assemble stiffness and mass matrix
[A,M,xx,yy] = assembleFEM(N); 

% setup exact, noisy data
ue = ub(2).*((xx-0.45).^2+(yy-0.55).^2<0.1) ...
     + (ub(3)-ub(2)).*((xx-0.4).^2+(yy-0.6).^2<0.02);
ye = A\(M*ue(:));
yd = ye(:) + delta*randn(size(ye))*max(ye); % noisy data 

tplot = @(n,f,s) tplot_(n,f,s,N,xx,yy);
dplot = @(n,f,s) dplot_(n,f,s,N,xx,yy,ub);

dplot(1,ue,'true parameter');
tplot(2,ye,'exact data');
tplot(3,yd,'noisy data');

% precompute some terms
Mz = M*yd(:);    AT = A;    N2 = N*N;    

%% compute reconstruction
% initialize iterates
y  = zeros(N2,1);                      % state variable
p  = zeros(N2,1);                      % dual variable
as = zeros(2*d,N2);                    % active sets

% continuation: start with sufficiently gamma^0
gamma = 1e-3;
while gamma > 1e-12
    fprintf('\nCompute solution for gamma = %1.3e:\n',gamma);
    it = 1;    nold = 1e99;    tau = 1;  tmin = 1e-9;
    ga = 1+2*gamma/alpha;
    while true
        % update active sets
        as_old = as;
        % Q_i^gamma
        as(1,:) = (p < alpha/2*(ga*ub(1)+ub(2)));
        for i = 2:d-1
            as(i,:) = (p > alpha/2*(ub(i-1)+ga*ub(i))) & ...
                      (p < alpha/2*(ga*ub(i)+ub(i+1)));
        end
        as(d,:) =  (p >= alpha/2*(ub(d-1)+ga*ub(d)));
        Hg = as(1:d,:)'*ub;
        % Q_i,i+1^gamma
        for i = 1:d-1
            ind = d+1+i;
                as(ind,:) = (p>=alpha/2*(ga*ub(i)+ub(i+1))) & ...
                            (p<=alpha/2*(ga*ub(i+1)+ub(i)));
                Hg  = Hg + (p-alpha/2*(ub(i)+ub(i+1))).*as(ind,:)'/gamma;
        end
        DHg = sum(as(d+2:2*d,:))/gamma;

        % system matrix, right hand side
        C   = [M AT; A -M*spdiags(DHg',0,N2,N2)];
        rhs = [Mz-M*y-AT*p; -A*y + M*Hg];
        nr  = norm(rhs(:));

        % line search
        if nr >= nold        % if no decrease: backtrack (never on first iteration)
            tau = tau/2;
            y = y - tau*dx(1:N*N);
            p = p - tau*dx(1+N*N:end);
            if tau < tmin   % terminate Newton iteration
                disp('step size too small')
                break;       
            else             % bypass rest of while loop; compute new gradient
                continue; 
            end
        end
        
        update = nnz((as-as_old));
        fprintf('%i\t%d\t\t%1.3e\t%1.3e\t%d\n',...
           it,update,nr,tau,sum(as(:)));
         if update == 0 && nr < 1e-6  % success, solution found
            break;
        elseif it == maxit            % failure, too many iterations
            break;
        end
        % otherwise update information, continue
        it = it+1;   nold = nr;   tau = 1;  
        % semismooth Newton step
        dx = C\rhs;
        y = y+dx(1:N2);
        p = p+dx(1+N2:end);        
    end   
    % check convergence
    if it < maxit                      % converged: accept iterate
        u = Hg;                  
        regnodes = nnz(as(d+2:end,:));
        fprintf('Solution has %i node(s) in regularized active sets\n',regnodes);
        if regnodes == 0               % solution optimal: terminate
            break;
        else                           % reduce gamma, continue
            gamma = gamma/2;
        end
    else                               % not converged: reject, terminate
        fprintf('Iterate rejected, returning u_gamma for gamma = %1.3e\n',gamma*10);
        break;
    end 
    dplot(99,u,'iterate')
end 

%% plot reconstruction
dplot(4,u,'reconstruction')

end % main function

function [K,M,xx,yy] = assembleFEM(n)
a   = 0;    b = 1;       % computational domain [a,b]^2
nel = 2*(n-1)^2;         % number of nodes
h2  = ((b-a)/(n-1))^2;   % Jacobi determinant of transformation (2*area(T))

% nodes
[xx,yy] = meshgrid(linspace(0,1,n));

% triangulation
tri = zeros(nel,3);
ind = 1;
for i = 1:n-1
    for j = 1:n-1
        node         = (i-1)*n+j+1;              % two triangles at node
        tri(ind,:)   = [node node-1 node+n];     % triangle 1 (lower left)
        tri(ind+1,:) = [node+n-1 node+n node-1]; % triangle 2 (upper right)
        ind = ind+2;
    end
end

% Mass and stiffness matrices
Ke = 1/2 * [2 -1 -1 -1 1 0 -1 0 1]';   % elemental stiffness matrix
Me = h2/24 * [2 1 1 1 2 1 1 1 2]';     % elemental mass matrix

ent = 9*nel;
row = zeros(ent,1);
col = zeros(ent,1);
valk = zeros(ent,1);
valm = zeros(ent,1);

ind = 1;
for el=1:nel
    ll       = ind:(ind+8);            % local node indices
    gl       = tri(el,:);              % global node indices
    row(ll)  = gl([1;1;1],:); rg = gl';
    col(ll)  = rg(:,[1 1 1]);
    valk(ll) = Ke;
    valm(ll) = Me;
    ind      = ind+9;
end
M = sparse(row,col,valm);
K = sparse(row,col,valk);

% modify matrices for homogenenous Dirichlet conditions
bdnod = [find(abs(xx-a) < eps); find(abs(yy-a) < eps); ...
         find(abs(xx-b) < eps); find(abs(yy-b) < eps)];
M(bdnod,:) = 0;
K(bdnod,:) = 0;  K(:,bdnod) = 0;
for j = bdnod'
    K(j,j) = 1; %#ok<SPRIX>
end
end

function tplot_(n,f,s,N,x,y)
figure(n); 
surf(x,y,reshape(f,N,N));
shading interp; lighting phong; camlight headlight; alpha(0.8);
title(s); xlabel('x_1'); ylabel('x_2');
drawnow;
end % tplot_ function

function dplot_(n,f,s,N,x,y,ub)
figure(n);
pcolor(x,y,reshape(f,N,N));
shading flat;
title(s); xlabel('x_1'); ylabel('x_2');
colorbar('Limits',[ub(1),0.15],'Ticks',ub); caxis([ub(1) 0.15]);drawnow;
end % dplot_ function
