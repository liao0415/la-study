clear;
%%setting
[node,elem] = squaremesh([0,1,0,1],0.25);
mesh = struct('node',node,'elem',elem);
option.L0 = 3;
option.maxIt = 4;
option.printlever = 1;

%% Non-empty Dirichlet boundary condition.
option.plotflag = 1;
pde = sincosdata;
fem_poisson(mesh,pde,option);

%% Pure Neumann boundary condition.
option.plotflag = 0;
pde = sincosNeumanndata;
% pde = sincosdata;
mesh.bdFlag = setboundary(node,elem,'Neumann');
fem_poisson(mesh,pde,option);