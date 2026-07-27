use pbr_proto::convert::{
    ConvertError, bin_config_to_edges, commitment_from_bytes, commitment_to_bytes,
    edges_to_bin_config, model_from_proto, model_to_proto, share_from_proto, share_to_proto,
    split_from_proto, split_to_proto,
};
use pbr_core::crypto::field::PRIME;
use pbr_core::{
    BinConfiguration, Commitment, Model, Share, SplitDecision, Tree, TreeNode, encode_all,
};

#[test]
fn share_roundtrips() {
    let share = Share {
        x: 2,
        values: encode_all(&[1.5, -3.25, 0.0]),
    };
    let proto = share_to_proto(&share);
    let back = share_from_proto(proto).unwrap();
    assert_eq!(back.x, 2);
    assert_eq!(back.values, share.values);
}

#[test]
fn oversized_field_element_rejected() {
    let proto = pbr_proto::v1::ShareVector {
        x: 1,
        values: vec![PRIME],
    };
    assert!(matches!(
        share_from_proto(proto),
        Err(ConvertError::FieldElementOutOfRange(_))
    ));
}

#[test]
fn zero_x_rejected() {
    let proto = pbr_proto::v1::ShareVector {
        x: 0,
        values: vec![42],
    };
    assert!(matches!(
        share_from_proto(proto),
        Err(ConvertError::InvalidEvaluationPoint)
    ));
}

#[test]
fn commitment_roundtrips_and_length_checked() {
    let c = Commitment([7u8; 32]);
    let bytes = commitment_to_bytes(&c);
    assert_eq!(commitment_from_bytes(&bytes).unwrap(), c);
    assert!(matches!(
        commitment_from_bytes(&[1u8; 31]),
        Err(ConvertError::BadCommitmentLength(31))
    ));
}

#[test]
fn bin_configuration_roundtrips() {
    let cfg = BinConfiguration {
        feature_idx: 3,
        edges: vec![0.0, 1.5, 2.75, 10.0],
    };
    let proto = bin_config_to_edges(&cfg);
    let back = edges_to_bin_config(proto);
    assert_eq!(back.feature_idx, cfg.feature_idx);
    assert_eq!(back.edges, cfg.edges);
}

#[test]
fn split_decision_roundtrips() {
    let split = SplitDecision {
        node_id: 2,
        feature_idx: 1,
        threshold: 0.42,
        gain: 1.23, // not carried by SplitNode; not roundtripped
        left_child_id: 3,
        right_child_id: 4,
        g_left: 5.0,  // not carried by SplitNode; not roundtripped
        h_left: 6.0,  // not carried by SplitNode; not roundtripped
        g_right: 7.0, // not carried by SplitNode; not roundtripped
        h_right: 8.0, // not carried by SplitNode; not roundtripped
    };
    let proto = split_to_proto(&split);
    // The node id travels as the splits_so_far map key, not in SplitNode.
    let back = split_from_proto(proto, split.node_id as u32);
    assert_eq!(back.node_id, split.node_id);
    assert_eq!(back.feature_idx, split.feature_idx);
    assert_eq!(back.threshold, split.threshold);
    assert_eq!(back.left_child_id, split.left_child_id);
    assert_eq!(back.right_child_id, split.right_child_id);
}

fn split(feature_idx: usize, threshold: f64, left: TreeNode, right: TreeNode) -> TreeNode {
    TreeNode::Split {
        feature_idx,
        threshold,
        gain: 0.0,
        left: Box::new(left),
        right: Box::new(right),
    }
}

fn leaf(value: f64) -> TreeNode {
    TreeNode::Leaf { value }
}

#[test]
fn model_roundtrips_prediction_equivalent() {
    let mut model = Model::new(0.3, 0.1);
    model.add_tree(Tree {
        root: split(0, 1.5, leaf(-0.2), leaf(0.4)),
    });
    // A deeper, left-heavy asymmetric tree, so the flatten/rebuild index
    // logic is exercised beyond the depth-1 case: pre-order emits the whole
    // left spine before any right child, giving non-adjacent child indices.
    model.add_tree(Tree {
        root: split(
            1,
            0.0,
            split(
                0,
                -1.0,
                split(2, 5.0, leaf(-0.7), split(1, -3.5, leaf(0.1), leaf(0.9))),
                leaf(0.3),
            ),
            split(2, 2.5, leaf(-0.1), leaf(0.6)),
        ),
    });

    let proto = model_to_proto(&model);
    let back = model_from_proto(proto).unwrap();

    let rows = vec![
        vec![0.0, -1.0, 6.0],
        vec![-2.0, -0.5, 4.0],
        vec![1.5, 0.5, 2.5],
        vec![3.0, 0.5, 3.0],
        vec![-1.5, -0.1, 5.5],
    ];
    assert_eq!(model.predict(&rows), back.predict(&rows));
}

fn model_of_nodes(nodes: Vec<pbr_proto::v1::TreeNodeProto>) -> pbr_proto::v1::ModelProto {
    pbr_proto::v1::ModelProto {
        initial_prediction: 0.0,
        learning_rate: 0.1,
        trees: vec![pbr_proto::v1::TreeProto { nodes }],
    }
}

fn leaf_proto(value: f64) -> pbr_proto::v1::TreeNodeProto {
    pbr_proto::v1::TreeNodeProto {
        node: Some(pbr_proto::v1::tree_node_proto::Node::Leaf(
            pbr_proto::v1::LeafProto { value },
        )),
    }
}

fn split_proto(left: u32, right: u32) -> pbr_proto::v1::TreeNodeProto {
    pbr_proto::v1::TreeNodeProto {
        node: Some(pbr_proto::v1::tree_node_proto::Node::Split(
            pbr_proto::v1::SplitProto {
                feature_idx: 0,
                threshold: 1.0,
                left,
                right,
            },
        )),
    }
}

#[test]
fn model_from_proto_rejects_missing_oneof() {
    let proto = model_of_nodes(vec![pbr_proto::v1::TreeNodeProto { node: None }]);
    assert!(matches!(
        model_from_proto(proto),
        Err(ConvertError::MissingField("node"))
    ));
}

#[test]
fn model_from_proto_rejects_oversized_tree() {
    // A structurally VALID left spine of 2049 splits (4099 nodes): split at
    // index 2i points left at the next split (2i+2) and right at a leaf
    // (2i+1), closing with a final leaf. Only the node-count cap rejects it
    // Without the cap it would decode into a depth-2049 Box chain whose
    // recursive Drop is the stack hazard the cap exists for.
    let k = 2049u32;
    let mut nodes = Vec::new();
    for i in 0..k {
        nodes.push(split_proto(2 * i + 2, 2 * i + 1));
        nodes.push(leaf_proto(0.0));
    }
    nodes.push(leaf_proto(0.0));
    assert!(matches!(
        model_from_proto(model_of_nodes(nodes)),
        Err(ConvertError::MalformedTree(_))
    ));
}

#[test]
fn model_from_proto_rejects_malformed_flat_trees() {
    // Every rejected shape: an empty node array, a child index past the end,
    // a self/backward reference (the cycle case), one node claimed by both
    // children, and a node no split references.
    let malformed = [
        vec![],
        vec![split_proto(1, 5), leaf_proto(0.1)],
        vec![split_proto(0, 1), leaf_proto(0.1)],
        vec![split_proto(1, 1), leaf_proto(0.1)],
        vec![leaf_proto(0.1), leaf_proto(0.2)],
    ];
    for nodes in malformed {
        assert!(
            matches!(
                model_from_proto(model_of_nodes(nodes.clone())),
                Err(ConvertError::MalformedTree(_))
            ),
            "expected MalformedTree for {nodes:?}"
        );
    }
}
