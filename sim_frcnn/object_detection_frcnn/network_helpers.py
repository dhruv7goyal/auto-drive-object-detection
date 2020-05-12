import torch


def test_copy_weights(m1, m2):
    """
    Tests that weights copied from m1 into m2, are actually refected in m2
    """
    m1_state_dict = m1.state_dict()
    m2_state_dict = m2.state_dict()
    weight_copy_flag = 1
    for name, param in m1_state_dict.items():
        if name in m2_state_dict:
            if not torch.all(torch.eq(param.data, m2_state_dict[name].data)):
                print("Something is incorrect for layer {} in 2nd model", name)
                weight_copy_flag = 0

    if weight_copy_flag:
        print('All is well')

    return 1


def copy_weights_between_models(m1, m2):
    """
    Copy weights for layers common between m1 and m2.
    From m1 => m2
    """

    # Load state dictionaries for m1 model and m2 model
    m1_state_dict = m1.state_dict()
    m2_state_dict = m2.state_dict()

    # Get m1 and m2 layer names
    m1_layer_names, m2_layer_names = [], []
    for name, param in m1_state_dict.items():
        m1_layer_names.append(name)
    for name, param in m2_state_dict.items():
        m2_layer_names.append(name)

    cnt = 0
    for ind in range(len(m1_layer_names)):
        if m1_layer_names[ind][:6] == 'resnet':
            cnt += 1
            m2_state_dict[m2_layer_names[ind]] = m1_state_dict[m1_layer_names[ind]].data

    m2.load_state_dict(m2_state_dict)

    print ('Count of layers whose weights were copied between two models', cnt)
    return m2


