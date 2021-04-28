module ParseRF
export co_occuring_feature_pairs


struct decision_tree
    features::Vector{Int64}
    tree::Dict{Int64,Vector{Int64}}
    leaf_idx::Vector{Bool}
    internal_node_features::Vector{Int64}
end


function re_index_tree(tree)
    d = Dict()
    for (key, value) in tree
        d[key + 1] = value .+ 1
    end
    return d
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

    try
        # start from first node and walk down the tree
        same_path = traverse_tree(feature_1_id, feature_2_id, tree)
    catch ex
        println(feature_pair)
        println(tree)
        throw(ex)
    end

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
    # tuples to vectors. Also sort at the same time
    feature_pairs = [sort([i for i in j]) for j in feature_pairs]
    reverse_feature_pairs = [reverse(j) for j in feature_pairs]
    
    # group as a julia decision_tree struct
    trees = [
        decision_tree(tree[1], re_index_tree(tree[2]), tree[3], tree[4]) 
        for tree in trees
    ]

    tree_idx_1 = relevant_trees(trees, feature_pairs)
    tree_idx_2 = relevant_trees(trees, reverse_feature_pairs)
    all_fp_tree_matches = merge(tree_idx_1, tree_idx_2) # combine
    
    # keys as tuples so can be converted back to python dict. vectors are 
    # converted to numpy arrays which are unhashable
    output = Dict()
    for (key, value) in all_fp_tree_matches
        output[Tuple(Int64(x) for x in key)] = value
    end

    return output
end

end
