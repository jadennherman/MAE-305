%% MAE 305 Project 2
% Numerical solution of Ax = b by six methods
% Textbook routines are from Esfandiari, 2nd edition, Chapter 4
% All functions are included at the end

clear;
clc;
format long g;

%% Given system
A = [3 -1  0  0  0;
     2  3 -1  0  0;
     0  2  3 -1  0;
     0  0  2  3 -1;
     0  0  0  2  3];

b = [0.1; 5; 3; 0; 5];

x0 = [1; 0; 1; 0; 1];
tol = 1e-10;
kmax = 1000;  % Jacobi requires more than the textbook default of 100

fprintf('\n============================================================\n');
fprintf('MAE 305 PROJECT 2\n');
fprintf('Tolerance for iterative methods: %.1e\n', tol);
fprintf('Initial vector: [%g %g %g %g %g]^T\n', x0);

%% (a) Gaussian elimination with partial pivoting and row scaling
fprintf('\n============================================================\n');
fprintf('GAUSSIAN ELIMINATION\n');
xGauss = GaussPivotScale(A,b);
printSolution('Gaussian elimination', xGauss, A, b);

%% (b) Doolittle method
fprintf('\n============================================================\n');
fprintf('DOOLITTLE METHOD\n');
[LDoolittle, UDoolittle] = DoolittleFactor(A);
xDoolittle = DoolittleMethod(A,b);
disp('L ='); disp(LDoolittle);
disp('U ='); disp(UDoolittle);
printSolution('Doolittle', xDoolittle, A, b);

%% (c) Cholesky method
% The textbook Cholesky method requires a symmetric positive-definite
% coefficient matrix. Check both requirements before using the method.
fprintf('\n============================================================\n');
fprintf('CHOLESKY METHOD\n');

choleskyValid = false;
if norm(A-A.',inf) >= 1e-12
    fprintf(['Cannot use the Cholesky method because the coefficient ', ...
             'matrix A is not symmetric.\n']);
else
    [~,positiveDefiniteFlag] = chol(A);
    if positiveDefiniteFlag ~= 0
        fprintf(['Cannot use the Cholesky method because the coefficient ', ...
                 'matrix A is not positive definite.\n']);
    else
        choleskyValid = true;
        [LCholesky, UCholesky] = CholeskyFactor(A);
        xCholesky = CholeskyMethod(A,b);
        disp('L ='); disp(LCholesky);
        disp('U = L'' ='); disp(UCholesky);
        printSolution('Cholesky', xCholesky, A, b);
    end
end

%% (d) Thomas method
fprintf('\n============================================================\n');
fprintf('THOMAS METHOD\n');
xThomas = ThomasMethod(A,b);
printSolution('Thomas', xThomas, A, b);

%% (e) Jacobi method
fprintf('\n============================================================\n');
fprintf('JACOBI METHOD\n');
[xJacobi, kJacobi, MJnorm, xJacobiHistory, errJacobi] = ...
    Jacobi(A,b,x0,tol,kmax);
fprintf('Infinity norm of Jacobi iteration matrix = %.12f\n', MJnorm);
fprintf('Iterations required = %d\n', kJacobi);
printSolution('Jacobi', xJacobi, A, b);
printIterationTable('Jacobi', xJacobiHistory, errJacobi);

%% (f) Gauss-Seidel method
fprintf('\n============================================================\n');
fprintf('GAUSS-SEIDEL METHOD\n');
[xGaussSeidel, kGaussSeidel, MGSnorm, xGSHistory, errGS] = ...
    GaussSeidel(A,b,x0,tol,kmax);
fprintf('Infinity norm of Gauss-Seidel iteration matrix = %.12f\n', ...
    MGSnorm);
fprintf('Iterations required = %d\n', kGaussSeidel);
printSolution('Gauss-Seidel', xGaussSeidel, A, b);
printIterationTable('Gauss-Seidel', xGSHistory, errGS);

%% Results summary
fprintf('\n============================================================\n');
fprintf('SUMMARY\n');
fprintf(['Method                         x1              x2              ', ...
         'x3              x4              x5          Residual inf-norm  Steps\n']);
printSummaryRow('Gaussian elimination',xGauss,A,b,0);
printSummaryRow('Doolittle',xDoolittle,A,b,0);
if choleskyValid
    printSummaryRow('Cholesky',xCholesky,A,b,0);
else
    fprintf('%-29s %s\n','Cholesky','Cannot use: A is not symmetric.');
end
printSummaryRow('Thomas',xThomas,A,b,0);
printSummaryRow('Jacobi',xJacobi,A,b,kJacobi);
printSummaryRow('Gauss-Seidel',xGaussSeidel,A,b,kGaussSeidel);

%% Textbook function Gaussian elimination
% Esfandiari, 2nd edition, page 101.
% Textbook correction: [~,I] is used so I is the row index returned by max.

function x = GaussPivotScale(A,b)
%
% GaussPivotScale uses Gauss elimination with partial pivoting and
% row scaling to solve the linear system Ax = b.
%
n = length(b);
A = [A b];       % Define augmented matrix
for k = 1:n-1,
    % Find maximum magnitude in each row
    M = max(abs(A(k:end, k:end-1)), [], 2);
    a = abs(A(k:end, k));       % Find maximum in kth column
    [~, I] = max(a./M);         % Find row with maximum ratio
    I = I + k - 1;              % Adjust relative row to actual row
    if I > k
        A([k I], :) = A([I k], :);    % Pivot rows
    end
    m = A(k+1:n, k)/A(k, k);           % Construct multipliers
    [Ak, M] = meshgrid(A(k, :), m);    % Create mesh
    A(k+1:n, :) = A(k+1:n, :) - Ak.*M;
end
Ab = A;
% Find the solution vector using back substitution
x = BackSub(Ab);
end

%% Textbook function BackSub
% Esfandiari, 2nd edition, page 101.

function x = BackSub(Ab)
%
% BackSub returns the solution vector of the upper triangular augmented
% matrix Ab using back substitution.
%
n = size(Ab, 1);
for k = n:-1:1,
    Ab(k, :) = Ab(k, :)./Ab(k, k);        % Construct multipliers
    Ab(1:k-1, n+1) = Ab(1:k-1, n+1) ...
        - Ab(1:k-1, k)*Ab(k, n+1);        % Adjust rows
end
x = Ab(:, end);
end

%% Textbook function DoolittleFactor
% Esfandiari, 2nd edition, page 110.

function [L, U] = DoolittleFactor(A)
%
% DoolittleFactor returns the Doolittle factorization of matrix A.
%
n = size(A,1);
L = eye(n); U = zeros(n,n);    % Initialize
for i = 1:n,
    U(i,i) = A(i,i)-L(i,1:i-1)*U(1:i-1,i);
    for j = i+1:n,
        U(i,j) = A(i,j)-L(i,1:i-1)*U(1:i-1,j);
        L(j,i) = (A(j,i)-L(j,1:i-1)*U(1:i-1,i))/U(i,i);
    end
end
end

%% Textbook function DoolittleMethod
% Esfandiari, 2nd edition, page 111.

function x = DoolittleMethod(A,b)
%
% DoolittleMethod uses the Doolittle factorization of matrix A and
% solves the ensuing triangular systems to find the solution vector x.
%
[L, U] = DoolittleFactor(A);      % Find Doolittle factorization of A
n = size(A,1);

% Solve the lower triangular system Ly = b (forward substitution)
y = zeros(n,1);
y(1) = b(1);
for i = 2:n,
    y(i) = b(i)-L(i,1:i-1)*y(1:i-1);
end

% Solve the upper triangular system Ux = y (back substitution)
x = zeros(n,1);
x(n) = y(n)/U(n,n);
for i = n-1:-1:1,
    x(i) = (y(i)-U(i,i+1:n)*x(i+1:n))/U(i,i);
end
end

%% Textbook function CholeskyFactor
% Esfandiari, 2nd edition, page 113.

function [L, U] = CholeskyFactor(A)
%
% CholeskyFactor returns the Cholesky factorization of matrix A.
%
n = size(A,1);
L = zeros(n,n);    % Initialize
for i = 1:n,
    L(i,i) = sqrt(A(i,i)-L(i,1:i-1)*L(i,1:i-1)');
    for j = i+1:n,
        L(j,i) = (A(j,i)-L(j,1:i-1)*L(i,1:i-1)')/L(i,i);
    end
end
U = L';
end

%% Textbook function CholeskyMethod
% Esfandiari, 2nd edition, pages 113-114.

function x = CholeskyMethod(A,b)
%
% CholeskyMethod uses the Cholesky factorization of matrix A and
% solves the ensuing triangular systems to find the solution vector x.
%
[L, U] = CholeskyFactor(A);       % Find Cholesky factorization of A
n = size(A,1);

% Solve the lower triangular system Ly = b (forward substitution)
y = zeros(n,1);
y(1) = b(1)/L(1,1);
for i = 2:n,
    y(i) = (b(i)-L(i,1:i-1)*y(1:i-1))/L(i,i);
end

% Solve the upper triangular system L'x = y (back substitution)
x = zeros(n,1);
x(n) = y(n)/U(n,n);
for i = n-1:-1:1,
    x(i) = (y(i)-U(i,i+1:n)*x(i+1:n))/U(i,i);
end
end

%% Textbook function ThomasMethod
% Esfandiari, 2nd edition, pages 105-106.

function x = ThomasMethod(A,b)
%
% ThomasMethod uses Thomas method to find the solution vector x of a
% tridiagonal system Ax = b.
%
n = size(A,1);
d = diag(A);         % Vector of diagonal entries of A
l = [0;diag(A,-1)];  % Vector of lower diagonal elements
u = [diag(A,1);0];   % Vector of upper diagonal elements

u(1) = u(1)/d(1); b(1) = b(1)/d(1);    % First equation

for i = 2:n-1,     % The next n-2 equations
    den = d(i) - u(i-1)*l(i);
    if den == 0,
        x = 'failure, division by zero';
        return
    end
    u(i) = u(i)/den;
    b(i) = (b(i)-b(i-1)*l(i))/den;
end

b(n) = (b(n)-b(n-1)*l(n))/(d(n)-u(n-1)*l(n)); % Last equation
x(n) = b(n);
for i = n-1:-1:1,
    x(i) = b(i) - u(i)*x(i+1);
end
x = x';
end

%% Textbook function Jacobi
% Esfandiari, 2nd edition, pages 123-124.
% Project additions: xHistory and errorHistory record the printed table.

function [x, k, MJnorm, xHistory, errorHistory] = ...
    Jacobi(A, b, x0, tol, kmax)
%
% Jacobi uses the Jacobi iteration method to approximate the solution
% of Ax = b.
%
if nargin < 3 || isempty(x0), x0 = zeros(size(b)); end
if nargin < 4 || isempty(tol), tol = 1e-4; end
if nargin < 5 || isempty(kmax), kmax = 100; end
x(:, 1) = x0;
D = diag(diag(A)); At = A - D;
L = tril(At); U = triu(At);
% Norm of Jacobi iteration matrix
M = -D\(L + U); MJnorm = norm(M, inf); B = D\b;
errorHistory = zeros(kmax,1);
% Perform iterations up to kmax
for k = 1:kmax,
    x(:, k+1) = M*x(:, k) + B; % Compute next approximation
    errorHistory(k) = norm(x(:, k+1) - x(:, k));
    if errorHistory(k) < tol, break; end  % Check convergence
end
xHistory = x(:,1:k+1);
errorHistory = errorHistory(1:k);
x = x(:, end);
end

%% Textbook function GaussSeidel
% Esfandiari, 2nd edition, pages 127-128.
% Project additions: xHistory and errorHistory record the printed table.

function [x, k, MGSnorm, xHistory, errorHistory] = ...
    GaussSeidel(A, b, x0, tol, kmax)
%
% GaussSeidel uses the Gauss-Seidel iteration method to approximate the
% solution of Ax = b.
%
if nargin < 3 || isempty(x0), x0 = zeros(size(b)); end
if nargin < 4 || isempty(tol), tol = 1e-4; end
if nargin < 5 || isempty(kmax), kmax = 100; end
x(:, 1) = x0;
D = diag(diag(A)); At = A - D;
L = tril(At); U = At - L;
% Norm of Gauss-Seidel iteration matrix
M = -(D + L)\U; MGSnorm = norm(M, inf); B = (D + L)\b;
errorHistory = zeros(kmax,1);
% Perform iterations up to kmax
for k = 1:kmax,
    x(:, k+1) = M*x(:, k) + B;
    errorHistory(k) = norm(x(:,k+1)-x(:, k));
    if errorHistory(k) < tol,
        break
    end
end
xHistory = x(:,1:k+1);
errorHistory = errorHistory(1:k);
x = x(:, end);
end

%% Project output helpers

function printSolution(methodName,x,A,b)
fprintf('%s solution:\n',methodName);
for i = 1:length(x)
    fprintf('x%d = %.12f\n',i,x(i));
end
fprintf('Infinity-norm residual = %.3e\n',norm(A*x-b,inf));
end

function printIterationTable(methodName,xHistory,errorHistory)
fprintf('\n%s iteration history\n',methodName);
fprintf(' k             x1             x2             x3             x4             x5        Step 2-norm\n');
fprintf('%3d %14.10f %14.10f %14.10f %14.10f %14.10f %18s\n', ...
    0,xHistory(:,1),'--');
for k = 1:length(errorHistory)
    fprintf('%3d %14.10f %14.10f %14.10f %14.10f %14.10f %14.3e\n', ...
        k,xHistory(:,k+1),errorHistory(k));
end
end

function printSummaryRow(methodName,x,A,b,steps)
fprintf('%-28s %14.10f %14.10f %14.10f %14.10f %14.10f %18.3e %6d\n', ...
    methodName,x,norm(A*x-b,inf),steps);
end