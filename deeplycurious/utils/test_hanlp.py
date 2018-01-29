# coding:utf-8
from jpype import *
startJVM(getDefaultJVMPath(), "-Djava.class.path=/home/xuhaowen/tools/hanlp/hanlp-1.3.2.jar:/home/xuhaowen/tools/hanlp", "-Xms1g", "-Xmx1g")
HanLP = JClass('com.hankcs.hanlp.HanLP')
print(HanLP.segment('你好,欢迎使用HanLP'))
NLPTokenizer = JClass('com.hankcs.hanlp.tokenizer.NLPTokenizer')
print(NLPTokenizer.segment('原告合一网络公司诉称：我公司享有电视剧《北京青年》（以下简称涉案电视剧）的信息网络传播权|经我公司调查发现，被告擅自在其经营的百度视频客户端上向公众播放上述作品，其行为侵犯了我公司享有的信息网络传播权经我公司调查发现，被告擅自在其经营的百度视频客户端上向公众播放上述作品，其行为侵犯了我公司享有的信息网络传播权|故诉至法院，请求法院判令百度公司：1、立即停止对涉案电视剧的播放行为|2012年5月7日，鑫宝源公司（授权人）向合一网络公司（被授权人）出具《授权书》，授权节目为授权人拥有合法版权的涉案电视剧，授权内容为信息网络传播权、制止侵权的权利及转授权的权利，授权性质为独占专有的信息网络传播权，授权期限为授权节目在中国大陆首轮卫视首轮上星播出之日起6年，授权区域为中国大陆|综上，百度公司的上述行为未经权利人许可，已构成侵权，应当依法承担停止侵权及赔偿损失的侵权责任|一、自本判决生效之日起，被告北京百度网讯科技有限公司停止涉案侵权行为'))
