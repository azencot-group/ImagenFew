import torch
from torch import nn


class LitEma(nn.Module):
    def __init__(self, model, decay=0.9999, use_num_upates=True,warmup=0):
        super().__init__()
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        self.m_name2s_name = {}
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        self.register_buffer('num_updates', torch.tensor(0,dtype=torch.int) if use_num_upates
                             else torch.tensor(-1,dtype=torch.int))

        for name, p in model.named_parameters():
            if p.requires_grad:
                #remove as '.'-character is not allowed in buffers
                s_name = name.replace('.','')
                self.m_name2s_name.update({name:s_name})
                self.register_buffer(s_name,p.clone().detach().data)

        self.collected_params = []
        self.warmup = warmup

    def forward(self,model):
        decay = self.decay

        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay,(1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            for key in m_param:
                if m_param[key].requires_grad:
                    sname = self.m_name2s_name[key]
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    if self.num_updates > self.warmup:
                        shadow_params[sname].sub_(one_minus_decay * (shadow_params[sname] - m_param[key]))
                    else:
                        shadow_params[sname].copy_(m_param[key])
                else:
                    assert not key in self.m_name2s_name

    def copy_to(self, model):
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad:
                m_param[key].data.copy_(shadow_params[self.m_name2s_name[key]].data)
            else:
                assert not key in self.m_name2s_name

    def store(self, parameters):
        """
        Save the current parameters for restoring later.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            temporarily stored.
        """
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the
        `copy_to` method. After validation (or model saving), use this to
        restore the former parameters.
        Args:
          parameters: Iterable of `torch.nn.Parameter`; the parameters to be
            updated with the stored parameters.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)
    def update_ema_params(self, model):
        """
        Update the EMA params to save params which are not already saved
        """
        m_param = dict(model.named_parameters())
        for key in m_param:
            if m_param[key].requires_grad:
                if key not in self.m_name2s_name:
                    s_name = key.replace('.','')
                    self.m_name2s_name.update({key:s_name})
                    self.register_buffer(s_name,m_param[key].clone().detach().data)




    def copy_to_no_gradient(self,model):
        """
        Copy the EMA parameters which DON'T have gradient, to the model. This is used when we fine tune only certain parts of the model
        Use this after freezing the needed layers
        """
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad or key not in self.m_name2s_name:
                continue
            sparam = shadow_params[self.m_name2s_name[key]].data
            m_param[key].data.copy_(sparam)

    def remove_no_gradient_params(self, model):
        """
        Remove the parameters with no gradient from the EMA model. This is used when we fine tune only certain parts of the model
        Use this after freezing the needed layers
        """
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())
        for key in m_param:
            if m_param[key].requires_grad or key not in self.m_name2s_name:
                continue
            delattr(self, self.m_name2s_name[key])
            del self.m_name2s_name[key]

    def reset_num_updates(self):
        self.num_updates.zero_()

    def setup_finetune(self,model):
        """
        This function performs 4 steps
        1. copy the parameters with no gradient, from the EMA model to the model
        2. remove the parameters with no gradient from the EMA model
        3. update the EMA model with the new parameters - if they are not already present
        4. zero the num_updates counter
        """
        self.copy_to(model)
        self.remove_no_gradient_params(model)
        # NOTE: this is not needed.
        # self.update_ema_params(model)
        self.reset_num_updates()









