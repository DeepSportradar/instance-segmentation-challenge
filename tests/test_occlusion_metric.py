from tools.occlusion_metric import *


PRED = torch.tensor([
    [1,0,1,0,2,0,2,0,3,0,3],
    [1,0,1,0,2,0,3,0,3,0,3],
    [1,0,1,0,2,0,0,0,0,0,0],
], dtype=torch.int32)
PRED_MASKS = torch.stack([PRED==i for i in range(0,4)],dim=0)
INSTANCES = torch.tensor([
    [1,0,1,0,2,0,2,0,3,0,3],
    [1,1,1,0,2,0,2,0,3,0,3],
    [1,0,1,0,2,0,0,0,0,3,3],
], dtype=torch.int32)
PARTS = torch.tensor([
    [2,0,2,0,3,0,1,0,4,0,4],
    [2,2,2,0,3,0,1,0,4,0,4],
    [2,0,2,0,3,0,0,0,0,4,4],
], dtype=torch.int32)


def test_parts():
    parts = compute_parts_2(INSTANCES)
    assert parts.equal(PARTS)
    part_size = compute_intersection(INSTANCES, parts)
    assert part_size[(1,2)] == 7
    assert part_size[(2,3)] == 3
    assert part_size[(2,1)] == 2
    assert part_size[(3,4)] == 6


def test_iou():
    iou1 = compute_iou(INSTANCES, PRED)
    iou2 = compute_iou_masks(INSTANCES, PRED_MASKS)
    assert iou1.sub(iou2).abs().max() < 1e-8
    assert iou1[1,1].sub(6/7).abs() < 1e-8
    assert iou1[2,2].sub(4/5).abs() < 1e-8
    assert iou1[3,3].sub(4/7).abs() < 1e-8


def test_matching():
    matching = compute_matching_masks(INSTANCES, PRED_MASKS)
    assert matching == ([(1,1),(2,2),(3,3)], set(), set())
    matching = compute_matching(PRED, PARTS)
    assert matching == ([(1,2),(2,3),(3,4)], set(), {1})


def test_om():
    om = compute_om(INSTANCES, PRED_MASKS, 1/torch.arange(1,PRED_MASKS.shape[0]+1))
    ip, it, pp, pt = om
    assert it == ip == 1
    assert pp == 1
    assert pt == 2

def test_complex_om():
    INSTANCES = torch.tensor([
        [1,0,1,0,2,0,2,0,3,0,3],
        [1,1,1,0,2,0,2,0,3,0,3],
        [1,0,1,0,2,0,3,0,0,3,3],
    ], dtype=torch.int32)
    PRED = torch.tensor([
        [1,0,1,0,2,0,2,0,3,0,3],
        [1,0,1,0,2,0,3,0,3,0,3],
        [1,0,1,0,2,0,0,0,0,0,0],
    ], dtype=torch.int32)
    PRED_MASKS = torch.stack([PRED==i for i in range(0,4)],dim=0)
    PRED_MASKS[2,2,6] = True
    PRED_MASKS[3,2,6] = True
    S = (1/torch.arange(PRED_MASKS.shape[0]))
    S[0] = 0
    om = compute_om(INSTANCES, PRED_MASKS, S)
    S = S.tolist()
    ip, it, pp, pt = om
    assert ip == 2
    assert it == 2
    assert pt == 3
    assert abs(pp - ( 1 + (S[3])/(S[3]+S[2]) )) < 1e-8
