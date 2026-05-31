import tensorflow as tf

class FocalLoss(tf.keras.losses.Loss):
    """
    gamma: quanto maior, mais foco em samples difíceis (padrão: 2.0)
    alpha: balanceamento de classes (padrão: 0.25)
    """
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        # Clip predictions para estabilidade
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        
        # Cross entropy
        ce = -y_true * tf.math.log(y_pred)
        
        # Focal weight: (1 - pt)^gamma
        # Samples com baixa probabilidade (difíceis) → peso alto
        # Samples com alta probabilidade (fáceis) → peso baixo
        focal_weight = tf.pow(1.0 - y_pred, self.gamma)
        
        # Apply alpha balancing
        focal_loss = self.alpha * focal_weight * ce
        
        return tf.reduce_mean(tf.reduce_sum(focal_loss, axis=1))
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha
        })
        return config