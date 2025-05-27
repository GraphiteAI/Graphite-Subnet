class SubnetPool():
    def __init__(self, num_tao_tokens, num_alpha_tokens, netuid):
        self.num_tao_tokens = num_tao_tokens
        self.num_alpha_tokens = num_alpha_tokens
        self.netuid = netuid

    # if the origin address is root pool or if the origin netuid is the same as the destination then charge the same
    def swap_alpha_to_tao(self, input_alpha_tokens):
        if self.netuid == 0:
            fee = 50000
            self.num_tao_tokens -= (input_alpha_tokens - fee)
            self.num_alpha_tokens += input_alpha_tokens
            return round(input_alpha_tokens - fee, 5)
        else:
            fee = input_alpha_tokens * 0.05/100 * (self.num_tao_tokens/self.num_alpha_tokens) # in terms of tao

            new_alpha = self.num_alpha_tokens + input_alpha_tokens
            new_tao = self.num_alpha_tokens*self.num_tao_tokens / new_alpha
            tao_emitted = self.num_tao_tokens - new_tao - fee

            self.num_tao_tokens = new_tao + fee
            self.num_alpha_tokens = new_alpha
            return round(tao_emitted, 5)

    def swap_tao_to_alpha(self, input_tao_tokens):
        if self.netuid == 0:
            fee = 50000
            self.num_tao_tokens += input_tao_tokens
            self.num_alpha_tokens -= (input_tao_tokens - fee)
            return round(input_tao_tokens - fee, 5)
        else:
            fee = 50000 # in terms of tao

            new_tao = self.num_tao_tokens + input_tao_tokens - fee
            new_alpha = self.num_alpha_tokens*self.num_tao_tokens / new_tao
            alpha_emitted = self.num_alpha_tokens - new_alpha

            self.num_tao_tokens = new_tao + fee
            self.num_alpha_tokens = new_alpha
            return round(alpha_emitted, 5)











