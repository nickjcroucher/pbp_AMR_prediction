module ParseRF
export co_occuring_feature_pairs, test


struct decision_tree
    features::Vector{Int64}
    tree::Dict{Int64,Vector{Int64}}
    leaf_idx::Vector{Bool}
    internal_node_features::Vector{Int64}
end


function traverse_tree(
    feature_1::Int64, 
    feature_2::Int64, 
    tree::decision_tree
)::Bool
    function recursive_search(node)
        children = tree.tree[node]
        if any([i == feature_2 for i in children])
            return true
        end
        if all([tree.leaf_idx[i] for i in children])
            return false
        end
        children = filter(x -> !tree.leaf_idx[x], children) # remove leaves
        return any([recursive_search(child) for child in children])
    end
    
    return recursive_search(feature_1)
end


function linked_features(
    tree::decision_tree, 
    feature_pair::Vector{Int64}, 
    both_permutations::Bool=false
)::Bool
    if !all([f in tree.internal_node_features for f in feature_pair])
        return false
    end

    feature_1_id = findall(tree.features .== feature_pair[1])[1]
    feature_2_id = findall(tree.features .== feature_pair[2])[1]

    # start from first node and walk down the tree
    same_path = traverse_tree(feature_1_id, feature_2_id, tree)
    if same_path
        return true
    end

    # start from second node and walk down the tree
    if both_permutations
        same_path = traverse_tree(feature_2_id, feature_1_id, tree)
        if same_path
            return true
        end
    end

    return false
end


function relevant_trees(
    trees::Vector{decision_tree}, 
    feature_pairs::Vector{Vector{Int64}}
)::Dict{Vector{Int64},Vector{Bool}}
    d = Dict()
    for fp in feature_pairs
        d[fp] = [linked_features(tree, fp) for tree in trees]
    end
    return d
end


function co_occuring_feature_pairs(
    trees,
    feature_pairs
)
    # pyjulia reads list of lists as vector of tuples, need to convert 
    # tuples to vectors
    feature_pairs = [[i for i in j] for j in feature_pairs]

    # group as a decision_tree object
    trees = [
        decision_tree(tree[1], tree[2], tree[3], tree[4]) for tree in trees
    ]

    for fp in feature_pairs
        sort!(fp)
    end
    tree_idx_1 = relevant_trees(trees, feature_pairs)

    for fp in feature_pairs
        reverse!(fp)
    end
    tree_idx_2 = relevant_trees(trees, feature_pairs)

    return vcat(tree_idx_1, tree_idx_2)
end

function test(a, b)
    println(typeof(a))
    println(typeof(b))
    sort!(b)
end

end