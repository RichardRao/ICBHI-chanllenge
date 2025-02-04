import torch

# moving average class
class MovingAverage:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new
    
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        n_cls = output.shape[1]
        valid_topk = [k for k in topk if k <= n_cls]
        
        maxk = max(valid_topk)
        bsz = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            if k in valid_topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / bsz))
            else: res.append(torch.tensor([0.]))

        return res, bsz
    
def update_moving_average(ma_beta, current_model, ma_ckpt):
    ma_updater = MovingAverage(ma_beta)
    new_state_dict = {}
    for (k1, current_params), (k2, ma_params) in zip(current_model.state_dict().items(), ma_ckpt.items()):
        assert k1 == k2
        old_weight, up_weight = ma_params.data, current_params.data
        new_state_dict[k1] = ma_updater.update_average(old_weight, up_weight)

    current_model.load_state_dict(new_state_dict)
    return current_model
    
def get_score(hits, counts, pflag=False):
    # normal accuracy
    sp = hits[0] / (counts[0] + 1e-10) * 100
    # abnormal accuracy
    se = sum(hits[1:]) / (sum(counts[1:]) + 1e-10) * 100
    sc = (sp + se) / 2.0

    if pflag:
        # print("************* Metrics ******************")
        print("S_p: {}, S_e: {}, Score: {}".format(sp, se, sc))

    return sp, se, sc