# 常用数学常量pi和对数函数
from math import pi, log

# 装饰器工具，用于保留被装饰函数的元信息
from functools import wraps

import torch

# nn——神经网络模块，包含各种层
# einsum——爱因斯坦求和约定，简洁表示张量乘法和广播运算
from torch import nn, einsum

# F——函数式接口，提供各种激活函数、卷积、损失函数等等
import torch.nn.functional as F

# rearrange——张量重排的高级接口
# repeat——重复操作的高级接口+
from einops import rearrange, repeat

# 高阶张量聚合（sum、mean等）层，可以直接嵌入nn.Module
from einops.layers.torch import Reduce



# 检查某个值是否存在
# val 存在返回 True，否则返回 False
def exists(val):
    return val is not None

# 如果 val 存在，返回 val 的值，否则返回 d 的值
def default(val, d):
    return val if exists(val) else d

'''
    装饰器
    用来缓存函数 f 的调用结果，避免对相同输入重复计算，从而提高性能
    输入：f
    输出：一个带缓存功能的函数 cached_fn
'''
def cache_fn(f):
    cache = dict()  # 创建一个空字典cache，存储结果

    # 使用wraps装饰内部函数 cached_fn
    # 功能：让被装饰的函数cached_fn还能保存原始函数 f 的元信息（如名称、文档字符串），方便调试与追踪
    @wraps(f)
    def cached_fn(*args, _cache = True, key = None, **kwargs):  # args——位置参数  _cache——是否启用缓存，默认启用  key——缓存的索引键（由调用者决定）  **kwargs——关键字参数
        # 如果 _cache 为false，则跳过缓存逻辑，直接执行原始函数，即直接返回结果，不存下来，
        if not _cache:
            return f(*args, **kwargs)

        # 否则——得缓存，只要不是已经存在的，就计算并缓存
        nonlocal cache # 告诉python，在该函数中使用外部作用域（cache_fn）的 cache 变量，即上面定义的字典cache
        # 如果缓存cache中已经存在相同的key（因为上面定义的cache为字典），那么直接返回之前的结果，即不重复计算
        if key in cache:
            return cache[key]
        # 否则——执行原始函数f，得到新的结果
        result = f(*args, **kwargs)
        # 存进缓存
        cache[key] = result
        return result   # 返回计算的结果

    return cached_fn    # 返回带缓存功能的包装函数

'''
    傅里叶特征编码fourier feature encoding
    功能：把连续变量（时间、空间坐标等）映射到高维周期空间，以便神经网络模型学习高频特征
    操作：通过对输入乘以一系列频率再取正弦和余弦，模型能更好地捕捉输入中的高频变化模式（例如细节、边界、周期性信息）
    输入：
        latent——张量
        max_freq——最大频率，用于控制编码范围
        num_bands——频率数量，也就是使用多少不同的频率进行编码，默认为4
    输出：经过Fourier编码后的张量，维度比原输入x更高，扩展为【sin，cos，原latent】
'''
def fourier_encode(x, max_freq, num_bands = 4):
    # 再最后再加一个维度，比如 本来是【batch，n】，变为【batch，n，1】
    x = x.unsqueeze(-1) # 方便后续与频率scales广播相乘

    # 保留原始的x的设备类型，数据类型，值
    device, dtype, orig_x = x.device, x.dtype, x

    # 生成一组等间隔的频率比例
    # 从 1.0 到 max_freq/2，一共num_bands个点
    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)

    # 形状匹配
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    # 对每个输入值乘以频率比例和pi
    # 这一步是为了给输入引入多个频率的周期性成分
    x = x * scales * pi

    # 对每个频率，生成sin和cos两种特征，把它们在最后一维拼接起来，使得每个原始维度被编码为多个周期信号
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    # 把原始输入拼回来
    x = torch.cat((x, orig_x), dim = -1)
    # 最后得到的输出：高频（sin，cos），原始值（低频部分）
    # 这样模型既能看到原始值，也能看到其高频
    return x


'''
    归一化
    功能：在调用fn之前，对输入和上下文进行归一化
    fn才是真正干事的模块，现在这个类只是在fn干活前，先归一化一下，方便fn干活
    即这个类其实是个准备工作类，让干活的工人吃饱饭，如果没有这个类，fn的输入可能太大太小，导致训练结果不稳定或者发散
'''
# 定义一个继承自 torch.nn.Module 的类 PreNorm
class PreNorm(nn.Module):
    # 初始化函数
    # dim——输入x的特征维度
    # fn——要被包装的函数
    # context_dim——如果该函数需要额外的上下文输入，就传入其维度
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()  # PyTorch模块初始化的标准写法
        self.fn = fn    # 保存传入的子模块（例如注意力层），等会在 forward 中调用
        self.norm = nn.LayerNorm(dim)   # 定义一个 Layer Normalization 层（归一化层），用于对输入 x 做归一化，使其特征分布更稳定，梯度更易传播

        # 如果存在 context_dim(即存在额外的上下文输入)，就创建一个对于的归一化层
        # 自注意力——只有x，x作为qkv
        # 交叉注意力——x作为q，context作为k和v
        # 否则 self.norm_context 为 None
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None  # 这里用到了上面定义的exists()函数

    # 前向传播时，先对输入 x 做归一化
    def forward(self, x, **kwargs):
        x = self.norm(x)

        # 如果存在norm_context，即该函数fn还需要额外输入context
        # 从kwargs里面取出context，归一化后，再放回去
        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        # 调用被包装的函数 fn ，输入为归一化后的 x 和可能被归一化的 context
        # 开始真正干活了
        return self.fn(x, **kwargs)

'''
    GEGLU——激活机制
    Gated GELU（门控线性单元，带GELU激活）
    作用：给输入加一个门。控制流动强度，从而让前馈层更加灵活稳定
    是前馈层的一种改进形式
    
    输入：张量latent
    输出：与latent一半维度相同的张量
    主要运算：把输入x一分为二，一份做数据，一份做门，然后用GELU激活们，最后二者再结合
    目的：增强非线性表达能量，提高训练稳定性
    
    发生——前馈层
    本质上是——特征选择
    不是注意力层，所以和我的动态掩码指导交叉注意力 不一样
    
'''
# 定义一个模块类
class GEGLU(nn.Module):
    # 定义前向传播逻辑
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1) # 在最后一个维度上分为两半
        # 对门信号进行GELU激活（一种平滑的非线性函数，比ReLU更柔和）
        # 逐元素相乘，相当于用门信号调节x的每个维度的强度
        return x * F.gelu(gates)
'''
问题：
1. 第二半的门控”是怎么工作的？
2. 为什么要分两半？
3. 为什么只对第二半做 GELU？

输入的x其实就是一个特意设计成两倍宽的向量
具体设计是对w权重矩阵进行设计，一半是负责提取语义，一半负责生成控制门信号
即W=【W1,W2】T，维度为2*dim，dim。
由于x为【batch_size,seq,dim】
那么最后得到的就是【batch_size,seq,2*dim】，上面的代码写的是在最后一个维度上分两半
上半截dim就是语义特征
下半截就是门控信号

问题4：为什么要这样写呢，不能直接在用个变量装x，然后对这个变量做门控，然后再乘回去？
即：
latent = some_input
gate = F.gelu(latent)   # 用同一个 x 生成门
y = latent * gate
问题在于：
1. 用同一套信息做门控。表达能力受限，模型无法学习那些信息该控制
2. 只是简单的缩放了

这样确实也能做门控，但是是为了门控而门控，忽视了真正的核心——模型真正要学的其实是 W 啊
而这个GEGLU则是为了 —— 学习门控信号且独立
'''


'''
    前馈神经网络，在每一个token上独立地做特征变换
    作用：
        对注意力层提取到的特征进行进一步建模
        通过非线性GEGLU和线性层增强模型的表达能力
        起到 信息混合 + 通道转换 的作用
    输入：latent 形状【batch_size, seq, dim】，即经过注意力层后的特征
    输出：与latent形状相同
'''
class FeedForward(nn.Module):
    # dim——输入 x 维度
    # mult——隐层扩张倍数，默认为4
    # dropout——防止过拟合，但这里默认为0，即不使用Dropout，不丢弃，完全保留原输出
    def __init__(self, dim, mult = 4, dropout = 0.):
        super().__init__()  # 初始化父类，标准写法

        # 这一层模块的核心
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2), # 扩展通道——把输入特征x的维度扩展到8倍，再乘个 2 是为了后面的GEGLU
            GEGLU(),    # 非线性门控激活——输出的最后一个维度回到 dim * mult
            nn.Linear(dim * mult, dim), # 压缩回原维度——再压回dim，这样得到的输出的形状就和输入x一样了
            nn.Dropout(dropout) # 随机丢弃部分神经元，防止过拟合
        )

    def forward(self, x):
        return self.net(x)  # 输入张量 x 直接通过 Sequential 中的层，形状不变


'''
    
    功能：计算输入注意力
    输入：latent--q, context(x)--k,v
    输出：
'''
class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head * heads    # 多头注意力展开后的总维度
        context_dim = default(context_dim, query_dim) # 如果context_dim未提供，即做的是自注意力，那么默认未query_dim，即latent的dim

        # scale 用于 scaled dot-product attention，防止内积过大导致 softmax 梯度消失
        self.scale = dim_head ** -0.5
        self.heads = heads

        # 生成Query
        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        # 一次性生成Key和Value，双通道，类似GEGLU的思路
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

        # 防止过拟合
        self.dropout = nn.Dropout(dropout)
        # 线性映射回原始Query维度
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads
        # q,k,v生成
        q = self.to_q(x)
        context = default(context, x) # 如果没有context，就自注意力，用参数x（即latent）
        k, v = self.to_kv(context).chunk(2, dim = -1) # 把key和value分开
        # 将 Q/K/V 按多头拆分，方便并行计算
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))
        # 计算 scaled dot-product 相似度矩阵
        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        # 动态掩码在这里生效
        # 未被 mask 的位置被置为极小值，softmax 后几乎为 0
        # 我的创新点的入口：用特征生成 mask 来指导 cross-attention
        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)    # 得到注意力权重
        attn = self.dropout(attn)   # 正则化，防止过拟合

        out = einsum('b i j, b j d -> b i d', attn, v)  # 注意力加权
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)   # 重构回原来的维度
        return self.to_out(out) # 线性映射回query_dim
'''
    我是希望以hsic-lasso得到所有数据的不同维度对最终分类标签的相关性权重alpha，然后不管是哪一batch，第一次的交叉注意力皆由alpha指导，即在最开始给latent划个重点
    我目前做的是在第一次时，直接乘上去了，但是当然，原始x是没有被改变的（等等，我也不确定，我得再看看我的代码）
    
    而后面的模型的交叉注意力学习，我是不用hsic-lasso的，而是用动态掩码
    动态掩码只是个称呼，具体的值是需要通过某些方法得到的
'''

# main class

class Perceiver(nn.Module):
    def __init__(
        self,
        *,
        num_freq_bands,
        depth,
        max_freq,
        input_channels = 3,
        input_axis = 2,
        num_latents = 512,
        latent_dim = 512,
        cross_heads = 1,
        latent_heads = 8,
        cross_dim_head = 64,
        latent_dim_head = 64,
        num_classes = 1000,
        attn_dropout = 0.,
        ff_dropout = 0.,
        weight_tie_layers = False,
        fourier_encode_data = True,
        self_per_cross_attn = 1,
        final_classifier_head = True
    ):
        """The shape of the final attention mechanism will be:
        depth * (cross attention -> self_per_cross_attn * self attention)

        Args:
          num_freq_bands: Number of freq bands, with original value (2 * K + 1)
          depth: Depth of net.
          max_freq: Maximum frequency, hyperparameter depending on how
              fine the data is.
          freq_base: Base for the frequency
          input_channels: Number of channels for each token of the input.
          input_axis: Number of axes for input data (2 for images, 3 for video)
          num_latents: Number of latents, or induced set points, or centroids.
              Different papers giving it different names.
          latent_dim: Latent dimension.
          cross_heads: Number of heads for cross attention. Paper said 1.
          latent_heads: Number of heads for latent self attention, 8.
          cross_dim_head: Number of dimensions per cross attention head.
          latent_dim_head: Number of dimensions per latent self attention head.
          num_classes: Output number of classes.
          attn_dropout: Attention dropout
          ff_dropout: Feedforward dropout
          weight_tie_layers: Whether to weight tie layers (optional).
          fourier_encode_data: Whether to auto-fourier encode the data, using
              the input_axis given. defaults to True, but can be turned off
              if you are fourier encoding the data yourself.
          self_per_cross_attn: Number of self attention blocks per cross attn.
          final_classifier_head: mean pool and project embeddings to number of classes (num_classes) at the end
        """
        super().__init__()
        self.input_axis = input_axis
        self.max_freq = max_freq
        self.num_freq_bands = num_freq_bands

        self.fourier_encode_data = fourier_encode_data
        fourier_channels = (input_axis * ((num_freq_bands * 2) + 1)) if fourier_encode_data else 0
        input_dim = fourier_channels + input_channels

        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        get_cross_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, input_dim, heads = cross_heads, dim_head = cross_dim_head, dropout = attn_dropout), context_dim = input_dim)
        get_cross_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))
        get_latent_attn = lambda: PreNorm(latent_dim, Attention(latent_dim, heads = latent_heads, dim_head = latent_dim_head, dropout = attn_dropout))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim, dropout = ff_dropout))

        get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff = map(cache_fn, (get_cross_attn, get_cross_ff, get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        for i in range(depth):
            should_cache = i > 0 and weight_tie_layers
            cache_args = {'_cache': should_cache}

            self_attns = nn.ModuleList([])

            for block_ind in range(self_per_cross_attn):
                self_attns.append(nn.ModuleList([
                    get_latent_attn(**cache_args, key = block_ind),
                    get_latent_ff(**cache_args, key = block_ind)
                ]))

            self.layers.append(nn.ModuleList([
                get_cross_attn(**cache_args),
                get_cross_ff(**cache_args),
                self_attns
            ]))

        self.to_logits = nn.Sequential(
            Reduce('b n d -> b d', 'mean'),
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, num_classes)
        ) if final_classifier_head else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        return_embeddings = False
    ):
        b, *axis, _, device, dtype = *data.shape, data.device, data.dtype
        assert len(axis) == self.input_axis, 'input data must have the right number of axis'

        if self.fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps=size, device=device, dtype=dtype), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos, indexing = 'ij'), dim = -1)
            enc_pos = fourier_encode(pos, self.max_freq, self.num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos), dim = -1)

        # concat to channels of data and flatten axis

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self.latents, 'n d -> b n d', b = b)

        # layers

        for cross_attn, cross_ff, self_attns in self.layers:
            x = cross_attn(x, context = data, mask = mask) + x
            x = cross_ff(x) + x

            for self_attn, self_ff in self_attns:
                x = self_attn(x) + x
                x = self_ff(x) + x

        # allow for fetching embeddings

        if return_embeddings:
            return x

        # to logits

        return self.to_logits(x)
