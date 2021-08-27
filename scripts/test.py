from diff_dalle import dist_util
import torch


def main():
    #args = create_argparser().parse_args()

    dist_util.setup_dist()
    mo = torch.randn(2,2, device=dist_util.dev())
    dist_util.sync_params(mo)


if __name__ == "__main__":
    main()
