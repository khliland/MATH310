function vmls_kmeans(X, k; maxiters = 100, tol = 5e-3)
    if ndims(X) == 2
        X = [X[i,:] for i in 1:size(X,1)]
    end;
    N = length(X)
    n = length(X[1])
    distances = zeros(N)
    reps = [zeros(n) for j=1:k]
    assignment = [ rand(1:k) for i in 1:N ]
    gs  = zeros(k,1);
    Jclust = []
    Jprevious = Inf
    for iter = 1:maxiters
        for j = 1:k
            group = [i for i=1:N if assignment[i] == j]
            #group = getindex.(findall(x -> x == j, assignment),1)
            gs[j] = length(group);
        if length(group)>0
            reps[j] = sum(X[group]) / gs[j];
        end
        end;
        for i = 1:N
            (distances[i], assignment[i]) =
                findmin([norm(X[i] - reps[j]) for j = 1:k])
        end;
        J = norm(distances)^2 /N
        Jclust      = vcat(Jclust,J);
        println("Iteration ", iter, ": Jclust = ", J, ".")
        if iter > 1 && abs(J - Jprevious) < tol * J
            return assignment, reps, Jclust, gs
        end
        Jprevious = J
    end
end
