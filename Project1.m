%% MAE 305 Project 1
% Numerical solution of x^2 - 5x + 4 = 0
% Using 2nd ed textbook for code format

clear;
clc;
format long g;

f = @(x)(x^2 - 5*x + 4);
tol = 1e-10;
kmax = 100;
trueRoots = [1, 4];

%% Bisection method
fprintf('\n============================================================\n');
fprintf('BISECTION METHOD\n');

fprintf('\nBracket for root 1: [-1.1, 2.5]\n');
[rootBis1, stepsBis1] = Bisection(f, -1.1, 2.5, kmax, tol);

fprintf('\nBracket for root 2: [2.0, 5.3]\n');
[rootBis2, stepsBis2] = Bisection(f, 2.0, 5.3, kmax, tol);

%% Regula falsi method
fprintf('\n============================================================\n');
fprintf('REGULA FALSI METHOD\n');

fprintf('\nBracket for root 1: [-1.1, 3.0]\n');
[rootRF1, stepsRF1] = RegulaFalsi(f, -1.1, 3.0, kmax, tol);

fprintf('\nBracket for root 2: [2.0, 6.0]\n');
[rootRF2, stepsRF2] = RegulaFalsi(f, 2.0, 6.0, kmax, tol);

%% Fixed-point method
%  Textbook Example 3.4.

fprintf('\n============================================================\n');
fprintf('FIXED-POINT METHOD\n');

g1 = @(x)((x^2 + 4)/5);
g2 = @(x)(5 - 4/x);

x1Small = 2.0;
x1Large = 3.0;

[rootFP1, stepsFP1] = FixedPoint(g1, x1Small, kmax, tol);
[rootFP2, stepsFP2] = FixedPoint(g2, x1Large, kmax, tol);

fprintf('g1 starting at %.1f converged to %.12f in %d steps.\n', ...
    x1Small, rootFP1, stepsFP1);
fprintf('g2 starting at %.1f converged to %.12f in %d steps.\n', ...
    x1Large, rootFP2, stepsFP2);

%% Newton-Raphson method
% The textbook Newton function accepts a symbolic function and uses
% matlabFunction to evaluate the function and its derivative

fprintf('\n============================================================\n');
fprintf('NEWTON-RAPHSON METHOD\n');

syms x;
fSymbolic = x^2 - 5*x + 4;

x1NewtonSmall = 0.0;
x1NewtonLarge = 5.0;

[rootNewton1, stepsNewton1] = Newton(fSymbolic, ...
    x1NewtonSmall, tol, kmax);
[rootNewton2, stepsNewton2] = Newton(fSymbolic, ...
    x1NewtonLarge, tol, kmax);

fprintf('Starting at %.1f converged to %.12f in %d steps.\n', ...
    x1NewtonSmall, rootNewton1, stepsNewton1);
fprintf('Starting at %.1f converged to %.12f in %d steps.\n', ...
    x1NewtonLarge, rootNewton2, stepsNewton2);

%% Results and absolute-error summary
fprintf('\n============================================================\n');
fprintf('SUMMARY (tolerance = %.1e)\n', tol);
fprintf(['Method             Root     Approximation          ', ...
    'Abs. error       Steps\n']);

fprintf('Bisection             1     %16.12f     %10.3e     %3d\n', ...
    rootBis1, abs(rootBis1 - trueRoots(1)), stepsBis1);
fprintf('Bisection             4     %16.12f     %10.3e     %3d\n', ...
    rootBis2, abs(rootBis2 - trueRoots(2)), stepsBis2);
fprintf('Regula falsi          1     %16.12f     %10.3e     %3d\n', ...
    rootRF1, abs(rootRF1 - trueRoots(1)), stepsRF1);
fprintf('Regula falsi          4     %16.12f     %10.3e     %3d\n', ...
    rootRF2, abs(rootRF2 - trueRoots(2)), stepsRF2);
fprintf('Fixed point g1        1     %16.12f     %10.3e     %3d\n', ...
    rootFP1, abs(rootFP1 - trueRoots(1)), stepsFP1);
fprintf('Fixed point g2        4     %16.12f     %10.3e     %3d\n', ...
    rootFP2, abs(rootFP2 - trueRoots(2)), stepsFP2);
fprintf('Newton-Raphson        1     %16.12f     %10.3e     %3d\n', ...
    rootNewton1, abs(rootNewton1 - trueRoots(1)), stepsNewton1);
fprintf('Newton-Raphson        4     %16.12f     %10.3e     %3d\n', ...
    rootNewton2, abs(rootNewton2 - trueRoots(2)), stepsNewton2);

%% Textbook function: Bisection
% Textbook page 59. The iteration count k was added as an output.

function [c, k] = Bisection(f, a, b, kmax, tol)

if nargin < 5 || isempty(tol), tol = 1e-4; end
if nargin < 4 || isempty(kmax), kmax = 20; end
if f(a)*f(b) > 0
    c = 'failure';
    k = 0;
    return
end
disp(' k      a      b     c     (b-a)/2')
for k = 1:kmax,
    c = (a+b)/2;     % Find the first midpoint
    if f(c) == 0,    % Stop if a root has been found
        return
    end
    fprintf('%3i     %11.6f%11.6f%11.6f%11.6f\n', ...
        k,a,b,c,(b-a)/2)
    if (b-a)/2 < tol,    % Stop if tolerance is met
        return
    end
    if f(b)*f(c) > 0    % Check sign changes
        b = c;            % Adjust the endpoint of interval
    else
        a = c;
    end
end

end

%% Textbook function: Regula Falsi
% Pages 62-63

function [r, k] = RegulaFalsi(f, a, b, kmax, tol)

if nargin < 5 || isempty(tol), tol = 1e-4; end
if nargin < 4 || isempty(kmax), kmax = 20; end
c = zeros(1,kmax);        % Pre-allocate
if f(a)*f(b) > 0
    r = 'failure';
    k = 0;
    return
end
disp(' k       a            b')
for k = 1:kmax,
    c(k) = (a*f(b)-b*f(a))/(f(b)-f(a));  % Find the x-intercept
    if f(c(k)) == 0       % Stop if a root has been found
        r = c(k);
        return
    end
    fprintf('%2i       %11.6f%11.6f\n',k,a,b)
    if f(b)*f(c(k)) > 0       % Check sign changes
        b = c(k);           % Adjust the endpoint of interval
    else
        a = c(k);
    end
    c(k+1) = (a*f(b)-b*f(a))/(f(b)-f(a)); % Find next x-intercept
    if abs(c(k+1)-c(k)) < tol,      % Stop if tolerance is met
        r = c(k+1);
        return
    end
end
r = 'failure';

end

%% Textbook function: Fixed Point
% Pages 67-68

function [r, n] = FixedPoint(g, x1, kmax, tol)

if nargin < 4 || isempty(tol), tol = 1e-4; end
if nargin < 3 || isempty(kmax), kmax = 20; end
x = zeros(1,kmax);
x(1) = x1;
for n = 1:kmax,
    x(n+1) = g(x(n));
    if abs(x(n+1) - x(n)) < tol
        r = x(n+1);
        return
    end
end
r = 'failure'; % Failure to converge after kmax iterations

end

%% Textbook function: Newton-Raphson
% Page 73

function [r, n] = Newton(f, x1, tol, N)

if nargin < 4 || isempty(N), N = 20; end
if nargin < 3 || isempty(tol), tol = 1e-4; end
% Find f' and convert to MATLAB function for evaluation
fp = matlabFunction(diff(f));
% Convert f to MATLAB function for evaluation
f = matlabFunction(f);
x = zeros(1, N+1);      % Pre-allocate
x(1) = x1;
for n = 1:N,
    if fp(x(n)) == 0,
        r = 'failure';
        return
    end
    x(n+1) = x(n) - f(x(n))/fp(x(n));
    if abs(x(n+1) - x(n)) < tol,
        r = x(n+1);
        return
    end
end
r = 'failure';

end
