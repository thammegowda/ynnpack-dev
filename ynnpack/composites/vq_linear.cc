#include "ynnpack/composites/vq_linear.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

namespace ynn {
namespace {

// Antipodal Gaussian VQ codebooks. Each table holds the 128 centroids c_k of a 256-codeword codebook {+c_k, -c_k}; a
// code byte is (sign << 7) | k. Block Hadamard rotation and per-group RMS normalization make weight vectors look like
// N(0, I) samples, so these universal tables serve every model, like the FP4 levels.
// Provenance: fit once, offline, with Lloyd's K-means algorithm restricted to antipodal codebooks on N(0, I_3) and
// N(0, I_4) samples. That run's seed and sample count were not recorded, so the digits cannot be regenerated exactly.
// vq_codebook.py implements the fit and verifies these tables: N(0, I) relative error is 20.6% (VQ3D) and 31.2%
// (VQ4D), each centroid lies within 0.03 of its Voronoi cell mean, and one more Lloyd step gains under 0.1%. The tables
// are part of the packed format; changing them changes every VQ weight and invalidates prepared checkpoints.
// Credits: the fit is the Lloyd (1957/1982) and Linde-Buzo-Gray (1980) vector quantizer design algorithm. Fitting
// codebooks to near-Gaussian weights after Hadamard rotation follows QuIP# (Tseng, Chee, Sun, Kuleshov and De Sa,
// arXiv:2402.04396) and QTIP (Tseng, Sun, Hou and De Sa, arXiv:2406.11235). Thanks also to kaitchup's chimachima
// Inkling-2-Mini quantization (huggingface.co/kaitchup/Inkling-2-Mini-chimachima-kaiPTQ), whose trellis-coded experts
// motivated moving from scalar formats to vector codes.
alignas(64) constexpr float kVq3dCodebook[128][4] = {
    {0.2898799F, 0.2189359F, 0.0749228F, 0.0000000F},    {0.0744203F, 0.0107628F, -0.3903789F, 0.0000000F},
    {-0.2483545F, 0.3149986F, -0.0951512F, 0.0000000F},  {-0.0811044F, 0.5000048F, 0.4001058F, 0.0000000F},
    {0.5954689F, -0.0634257F, -0.3119970F, 0.0000000F},  {0.3207588F, 0.5084633F, -0.3685102F, 0.0000000F},
    {-0.1713647F, 0.3790799F, -0.6152509F, 0.0000000F},  {0.4556742F, 0.2283692F, 0.5745244F, 0.0000000F},
    {-0.0709471F, 0.8365405F, -0.1018024F, 0.0000000F},  {0.7533528F, -0.2242074F, 0.3436033F, 0.0000000F},
    {-0.8415058F, -0.2768689F, -0.0771307F, 0.0000000F}, {-0.3805988F, 0.1845892F, 0.7844494F, 0.0000000F},
    {0.4901944F, 0.7518770F, 0.1277227F, 0.0000000F},    {0.6673278F, -0.6213187F, -0.0701191F, 0.0000000F},
    {0.1476153F, 0.0771828F, 0.9604355F, 0.0000000F},    {-0.5213371F, 0.7543382F, -0.4957215F, 0.0000000F},
    {-0.3656483F, -0.3552157F, 0.9180980F, 0.0000000F},  {-0.8234191F, -0.4260606F, 0.5992307F, 0.0000000F},
    {0.1909407F, 0.9394469F, 0.5769568F, 0.0000000F},    {-0.3528909F, 1.0217944F, 0.3818939F, 0.0000000F},
    {1.1186795F, -0.1706294F, -0.2882442F, 0.0000000F},  {-0.6031238F, 0.3315857F, -0.9488689F, 0.0000000F},
    {-0.7798422F, 0.5841293F, 0.6531531F, 0.0000000F},   {0.2200582F, -0.7126142F, -0.9130256F, 0.0000000F},
    {0.4305419F, 1.0263374F, -0.4630372F, 0.0000000F},   {0.7275026F, 0.6876300F, 0.6986957F, 0.0000000F},
    {-0.0230503F, 0.8707001F, -0.9098471F, 0.0000000F},  {1.0628237F, 0.7490813F, -0.1980066F, 0.0000000F},
    {1.1646001F, 0.1550961F, 0.6158078F, 0.0000000F},    {-0.2990789F, -0.5928549F, -1.1590384F, 0.0000000F},
    {0.9063085F, -0.0699866F, -1.0281837F, 0.0000000F},  {-1.3043009F, 0.4167053F, -0.1957438F, 0.0000000F},
    {0.1908410F, 1.3854651F, 0.0048890F, 0.0000000F},    {-0.1079475F, 0.4325555F, -1.3333783F, 0.0000000F},
    {0.2143581F, -1.2980422F, 0.5376550F, 0.0000000F},   {0.8445613F, 0.2078320F, 1.1287910F, 0.0000000F},
    {0.2198870F, -0.1848553F, -1.3992769F, 0.0000000F},  {0.6028414F, -1.3128448F, 0.0715116F, 0.0000000F},
    {1.1716700F, 0.7911257F, 0.4094561F, 0.0000000F},    {1.0432485F, -0.9602825F, 0.4579244F, 0.0000000F},
    {1.4679375F, 0.2876970F, 0.0675683F, 0.0000000F},    {1.1263916F, -0.9586329F, -0.2561364F, 0.0000000F},
    {0.8350733F, 1.2828251F, 0.1200522F, 0.0000000F},    {1.1777767F, -0.4633834F, 0.8774280F, 0.0000000F},
    {-1.4204741F, -0.2593929F, 0.5938943F, 0.0000000F},  {0.5550837F, 0.9257753F, -1.1631069F, 0.0000000F},
    {0.6173332F, -0.6975386F, -1.3161926F, 0.0000000F},  {-0.6683343F, 1.0194820F, -1.0694693F, 0.0000000F},
    {-0.5813711F, -1.3828474F, -0.6833405F, 0.0000000F}, {1.0088800F, 1.1547633F, -0.6341630F, 0.0000000F},
    {0.7274731F, -1.2649899F, -0.7950322F, 0.0000000F},  {-0.5690567F, -0.3214289F, 1.5462452F, 0.0000000F},
    {-0.4368663F, -0.0419634F, -1.6297680F, 0.0000000F}, {-1.4423393F, 0.4403062F, 0.7585111F, 0.0000000F},
    {0.1299101F, -1.6053116F, -0.5261052F, 0.0000000F},  {0.0199139F, -1.2481427F, -1.1963667F, 0.0000000F},
    {1.1567428F, 0.5790309F, -1.1825204F, 0.0000000F},   {0.7635897F, 1.0317841F, 1.2668504F, 0.0000000F},
    {0.3650476F, 1.5453492F, -0.8865762F, 0.0000000F},   {0.8128211F, -0.5436440F, 1.5877069F, 0.0000000F},
    {-0.0449841F, -0.6772541F, -1.8245056F, 0.0000000F}, {-1.4227908F, -0.6214433F, -1.1797643F, 0.0000000F},
    {-1.9448340F, 0.2021910F, 0.0671746F, 0.0000000F},   {0.0851318F, -0.9588928F, 1.7034483F, 0.0000000F},
    {1.8130980F, -0.1167096F, 0.7292802F, 0.0000000F},   {-1.2200382F, 0.9635012F, 1.2066981F, 0.0000000F},
    {0.0768598F, -1.9637331F, 0.3506002F, 0.0000000F},   {0.8360032F, 1.8040820F, -0.3911012F, 0.0000000F},
    {1.5936424F, 1.2682247F, -0.0436501F, 0.0000000F},   {0.7541385F, -0.3266100F, -1.8662587F, 0.0000000F},
    {-1.6614271F, -0.9336444F, 0.7314097F, 0.0000000F},  {-0.2698952F, 1.6507722F, -1.1742766F, 0.0000000F},
    {-1.8111163F, 0.8874663F, 0.3616909F, 0.0000000F},   {-1.8103963F, 0.8803040F, -0.4231153F, 0.0000000F},
    {-1.8639493F, -0.6784300F, -0.5676723F, 0.0000000F}, {-0.7126796F, 1.9300449F, 0.1982470F, 0.0000000F},
    {1.3913200F, -1.5249757F, 0.1914246F, 0.0000000F},   {1.4005505F, -0.1054376F, -1.5341072F, 0.0000000F},
    {-0.8737168F, 1.7429576F, -0.7338665F, 0.0000000F},  {1.4597447F, -0.1450316F, 1.4807189F, 0.0000000F},
    {-1.3476816F, -1.3652819F, -0.8340404F, 0.0000000F}, {0.9009436F, 0.5237538F, 1.8313551F, 0.0000000F},
    {0.3657959F, 2.0823479F, 0.3175313F, 0.0000000F},    {-1.4848330F, 1.0785228F, -1.1306248F, 0.0000000F},
    {2.0858359F, 0.5869824F, -0.1928570F, 0.0000000F},   {-0.0669705F, -0.1845283F, 2.1908808F, 0.0000000F},
    {-2.0186257F, -0.0266904F, 0.9492834F, 0.0000000F},  {1.3981714F, -1.6016669F, -0.6760171F, 0.0000000F},
    {0.5254046F, -1.8894621F, -1.1149150F, 0.0000000F},  {0.8144223F, 0.8327990F, -1.9422227F, 0.0000000F},
    {1.1677659F, 1.4709561F, -1.2748822F, 0.0000000F},   {-0.5796886F, 1.3184586F, 1.7852290F, 0.0000000F},
    {-0.4660698F, -1.9432108F, -1.1712487F, 0.0000000F}, {1.2377568F, 1.9695584F, 0.3160846F, 0.0000000F},
    {-0.3427452F, -1.6156869F, 1.7378801F, 0.0000000F},  {0.5213658F, 1.3664385F, 1.9341550F, 0.0000000F},
    {0.9132584F, -1.3684654F, 1.8097477F, 0.0000000F},   {-0.8569364F, 0.3881720F, -2.3874452F, 0.0000000F},
    {1.8167614F, 0.8262727F, -1.6488230F, 0.0000000F},   {2.1043532F, -1.0641659F, -1.2190106F, 0.0000000F},
    {0.5973481F, 2.3895240F, -1.1391423F, 0.0000000F},   {-1.5652775F, -1.4378489F, -1.7352074F, 0.0000000F},
    {-1.4475120F, 0.9566346F, 2.1260912F, 0.0000000F},   {-2.7163582F, -0.2742751F, -0.3494767F, 0.0000000F},
    {2.3370097F, 0.5258971F, 1.3655640F, 0.0000000F},    {-0.4381433F, 0.7981559F, 2.6497097F, 0.0000000F},
    {-2.5235462F, 0.7302494F, -0.9739844F, 0.0000000F},  {-0.5332519F, 2.7005503F, -0.5463765F, 0.0000000F},
    {-0.2985675F, 2.7094035F, 0.7202008F, 0.0000000F},   {1.8295492F, 2.0474744F, -0.6615931F, 0.0000000F},
    {2.3335645F, -1.6117210F, -0.0549791F, 0.0000000F},  {1.0983748F, 0.2349921F, -2.6094096F, 0.0000000F},
    {0.6111879F, 0.6429510F, 2.7387154F, 0.0000000F},    {1.8682412F, -0.8898856F, 2.0037427F, 0.0000000F},
    {1.4897261F, -2.4642906F, -0.0771507F, 0.0000000F},  {-2.3589020F, -1.5707153F, -0.5369092F, 0.0000000F},
    {2.7997298F, -0.5070987F, -0.5026782F, 0.0000000F},  {-1.7374908F, -0.3598339F, -2.2850394F, 0.0000000F},
    {-0.7323428F, -2.8136532F, 0.1813731F, 0.0000000F},  {-1.7401024F, 2.0260928F, -1.1848994F, 0.0000000F},
    {0.0830396F, -1.1715215F, 2.6970251F, 0.0000000F},   {0.5602890F, -2.2985246F, 1.7523545F, 0.0000000F},
    {2.6419337F, 1.0173181F, -0.8396603F, 0.0000000F},   {1.4341725F, -2.1126952F, -1.5036607F, 0.0000000F},
    {2.3243022F, -0.1077049F, -1.8656527F, 0.0000000F},  {0.0082220F, -2.1909935F, -2.1079016F, 0.0000000F},
    {1.3015021F, 2.5129416F, 1.1524514F, 0.0000000F},    {1.3647493F, 1.7584409F, -2.2069926F, 0.0000000F},
};

alignas(64) constexpr float kVq4dCodebook[128][4] = {
    {0.0437200F, 0.1217694F, -0.0441143F, 0.4343892F},    {-0.5248600F, 0.1163844F, 0.5267759F, 0.2263997F},
    {0.2761841F, -0.2451668F, 0.7518198F, -0.0275450F},   {0.2708326F, 0.7097974F, -0.3175223F, -0.1690759F},
    {-0.8697510F, -0.1787795F, -0.0853521F, -0.2343484F}, {0.5120996F, -0.7654647F, -0.0452275F, 0.1022875F},
    {0.3287048F, 0.7041430F, 0.5277953F, -0.0243743F},    {-0.6355621F, -0.4089907F, -0.1386092F, 0.6843871F},
    {-0.1342500F, 0.4865970F, 0.6021732F, -0.6770990F},   {-0.1355040F, 0.3420413F, -0.5262909F, -0.8249910F},
    {0.9269828F, -0.3508645F, 0.3065272F, -0.3787236F},   {-0.5340373F, -0.4253886F, 0.7309237F, -0.4996641F},
    {-0.1831310F, 0.6929705F, 0.5491363F, 0.6719608F},    {-0.8004854F, 0.3644387F, 0.0246176F, -0.8548269F},
    {0.2209345F, -0.1338705F, 0.8040862F, -0.9173695F},   {0.4803013F, 0.2921601F, 0.9446446F, 0.5983542F},
    {-0.2823305F, 0.9103254F, -0.4640681F, 0.6986114F},   {0.6440857F, 1.0569475F, 0.0170087F, 0.5683000F},
    {-0.3531377F, 0.2634634F, 1.2935088F, -0.0511153F},   {-0.0651832F, -0.5780359F, -0.0963903F, -1.3336190F},
    {-0.8270944F, 0.3978502F, 0.1133818F, 1.1496739F},    {-1.2146748F, -0.4924251F, 0.6398579F, 0.3770189F},
    {-1.0324554F, 0.5133782F, 0.8319780F, -0.5183606F},   {-0.0197338F, 1.3168741F, 0.1593540F, -0.7537091F},
    {0.1869119F, 0.3534242F, 0.0872103F, -1.4713310F},    {1.1685336F, 0.3734141F, 0.8897201F, -0.3709285F},
    {0.5780652F, 0.5130802F, -1.3033636F, -0.3778219F},   {0.8836918F, 0.5001590F, 0.1735313F, 1.1812068F},
    {-0.0623595F, 1.1302764F, -1.0867079F, -0.2009189F},  {0.6412075F, 0.0303191F, -0.9313703F, -1.1232985F},
    {1.0472827F, -0.4253230F, 1.0989852F, 0.2464536F},    {0.2474827F, -1.2413141F, -0.9570146F, 0.1806056F},
    {0.1460970F, 1.5687497F, -0.3206204F, 0.0356505F},    {1.0615031F, 1.1819384F, 0.0609634F, -0.3442878F},
    {0.4184644F, 0.3142740F, 1.5042356F, -0.3579919F},    {0.8877898F, -0.9370761F, -0.9553580F, -0.4840269F},
    {-1.4648855F, 0.3849353F, 0.6129368F, 0.3862399F},    {-0.5216316F, -1.1186372F, 0.6109977F, 1.0077583F},
    {-1.5556110F, 0.5813699F, -0.2074663F, -0.3599600F},  {-1.3323330F, -0.7477430F, -0.6143458F, -0.4785758F},
    {-0.6802459F, 1.1926585F, -0.5693121F, -0.8639194F},  {0.8903326F, -1.2930937F, 0.6947021F, -0.1074649F},
    {0.5940695F, 0.1666665F, -0.7831483F, 1.4057575F},    {-1.4934007F, -0.2783388F, 0.5419408F, -0.6431727F},
    {-0.3407581F, 1.5776056F, 0.2199408F, 0.6107376F},    {-0.3146284F, 0.4023769F, -1.5480362F, -0.6624789F},
    {-1.7300264F, -0.1992751F, -0.2066855F, 0.2660571F},  {0.9270948F, -0.5121868F, 1.2137754F, -0.7815127F},
    {0.3995820F, 1.2138886F, 1.0628878F, 0.7144401F},     {-1.0173453F, -1.1702085F, 0.9138098F, -0.2064368F},
    {-0.7468279F, -1.2124676F, -1.0060982F, 0.5528879F},  {-1.2414954F, 1.1718230F, -0.1487758F, 0.6909458F},
    {-0.7942368F, -0.4710005F, -0.9093923F, 1.3449353F},  {-0.6862920F, 0.9424152F, 0.5160111F, -1.3658235F},
    {-1.2398806F, 0.1917316F, -0.8298721F, -1.1279064F},  {0.0796766F, -0.7668836F, 1.6046135F, -0.6197022F},
    {1.2497897F, 0.5870848F, -0.0988800F, -1.2858201F},   {-1.0944479F, 1.4804146F, 0.3915898F, -0.3222347F},
    {-0.4390914F, -1.2773576F, 0.6768451F, -1.1919886F},  {0.2247807F, -0.7521216F, -1.3975850F, 1.1112111F},
    {-0.5716707F, 0.3184846F, -0.1829846F, -1.8379883F},  {-0.1430758F, 0.6742385F, -1.0031463F, -1.5679462F},
    {0.3302614F, 0.4922758F, 1.0734009F, 1.5828937F},     {-0.8110709F, -0.3672691F, 1.6489997F, -0.7277758F},
    {-0.2835191F, 0.5266535F, 1.8443546F, 0.6549368F},    {0.3914427F, -0.6166486F, 0.7620581F, -1.7584860F},
    {-1.6137120F, 0.3629365F, -0.4755709F, 1.2049598F},   {0.6106057F, 1.9708486F, 0.5323983F, 0.2202234F},
    {-0.4376008F, -1.3452383F, -0.4554858F, 1.6842543F},  {0.6543137F, -1.1636745F, -0.9316387F, -1.5485293F},
    {0.9886925F, 0.7197132F, 1.8489975F, 0.3944862F},     {-1.5071229F, 0.0911527F, 1.7649119F, -0.0696960F},
    {0.2239597F, -1.8891914F, 1.1394593F, -0.7093123F},   {2.0569181F, 1.2657236F, -0.0030667F, 0.1934151F},
    {-0.5955182F, 0.2799602F, 0.4511086F, 2.3106191F},    {0.4898651F, 1.7698807F, 0.4715228F, 1.5695323F},
    {-1.5751944F, -1.2839772F, 0.2124300F, -1.3733664F},  {1.5981737F, 0.0655479F, -0.3520485F, 1.8396319F},
    {1.0849370F, -1.2882707F, -1.8267192F, 0.2275232F},   {1.9932755F, -0.7560778F, 1.2553840F, -0.3746380F},
    {-0.3666895F, -2.2720835F, 0.7546551F, 0.6348566F},   {2.0878565F, 0.2473644F, 1.3632013F, 0.3099209F},
    {-0.8137578F, 1.5351169F, -0.1717016F, 1.8241254F},   {-1.6455512F, 1.1671102F, -0.0502905F, -1.5225588F},
    {1.4525228F, 0.7115986F, -1.7138934F, -0.9343363F},   {-0.2149620F, 0.1258112F, -1.8997753F, 1.6709995F},
    {-1.8265983F, 0.2055664F, 0.7232771F, 1.6137468F},    {-0.0216137F, -2.3388271F, -0.7104666F, 0.7310564F},
    {-0.2659459F, -1.6027545F, -1.9582624F, 0.2044537F},  {-1.0036333F, 0.3808315F, -2.3179610F, 0.1612135F},
    {-2.1528568F, 0.9293532F, 0.9020888F, -0.5181557F},   {0.8986555F, -2.3355966F, 0.4102524F, 0.3920203F},
    {-2.3667645F, -0.4140102F, 0.8980856F, 0.2064885F},   {-0.3246169F, -0.5872661F, 2.4672444F, 0.3050033F},
    {-0.8006525F, 1.6434002F, -1.8080813F, -0.1406017F},  {-1.1749686F, -2.1945264F, 0.4565000F, -0.5152256F},
    {-0.1532861F, -1.5078635F, 1.6976945F, 1.2405920F},   {1.3411070F, -0.4462811F, -1.5595809F, 1.5349046F},
    {1.5130712F, -0.7613766F, -1.7184885F, -0.9828388F},  {0.2225349F, -0.4221108F, -2.5067046F, 0.5457271F},
    {-1.6734394F, -1.5824093F, 0.8721709F, 0.8675566F},   {2.4329751F, 0.0293003F, 0.1957903F, 0.9291463F},
    {-0.5279529F, -0.8605753F, 0.1264233F, -2.4072964F},  {1.3051212F, 1.9986709F, 0.3663986F, -1.0243680F},
    {1.4946986F, -1.1165807F, 1.4745142F, 1.1245606F},    {1.5824517F, 0.4243353F, 1.8260409F, -0.9736202F},
    {2.0234179F, -1.5966105F, 0.5185172F, 0.2412388F},    {1.5724102F, 0.9044221F, 1.0734413F, 1.5975677F},
    {0.9187080F, 1.0560505F, -0.6110905F, -2.1685481F},   {-0.3590648F, 1.8410856F, -0.3740296F, -1.8404455F},
    {-2.5031071F, 0.6727738F, 0.0431494F, 0.5933259F},    {-2.2270184F, -0.8870912F, -0.5039949F, 1.0416789F},
    {0.5953999F, -2.2560027F, -1.1918265F, -0.4975150F},  {-0.6208572F, -0.0685742F, 1.8174524F, 1.8511490F},
    {0.1859279F, -0.5457648F, -0.8105114F, 2.4859238F},   {-1.7258431F, 1.7853631F, 0.5671817F, 0.8375367F},
    {0.8046500F, 1.6803378F, -1.9285091F, 0.1851606F},    {0.8620370F, -0.0284108F, 2.1562133F, 1.3693364F},
    {-1.5945938F, -1.7734820F, -1.2307477F, -0.2815099F}, {-1.2335289F, 1.3971338F, -1.4139540F, 1.3709079F},
    {-1.8771785F, -0.9424658F, 1.4154423F, -1.0042686F},  {0.9672250F, -1.8595692F, -1.1337005F, 1.3327132F},
    {0.2509387F, -2.5737910F, 0.2496832F, -0.8986252F},   {-1.4145545F, -0.0370440F, -0.7438263F, 2.2738602F},
    {-0.5647665F, -1.2725545F, 1.6059968F, -1.8082711F},  {0.6883293F, 1.4159659F, 1.7395996F, -1.5814461F},
    {-0.1017866F, -1.4694790F, -1.9894792F, -1.3993969F}, {-0.9326470F, 0.4084964F, -1.0097548F, -2.5033813F},
};

struct SoaCodebook {
    alignas(64) float c0[128];
    alignas(64) float c1[128];
    alignas(64) float c2[128];
    alignas(64) float c3[128];
    alignas(64) float half_norm2[128];
};

SoaCodebook build_soa(const float (*table)[4]) {
    SoaCodebook soa{};
    for (int i = 0; i < 128; ++i) {
        soa.c0[i] = table[i][0];
        soa.c1[i] = table[i][1];
        soa.c2[i] = table[i][2];
        soa.c3[i] = table[i][3];
        soa.half_norm2[i] = 0.5F * (table[i][0] * table[i][0] + table[i][1] * table[i][1] + table[i][2] * table[i][2] +
                                    table[i][3] * table[i][3]);
    }
    return soa;
}

const SoaCodebook& soa_codebook(int32_t dim) {
    static const SoaCodebook kSoa3 = build_soa(kVq3dCodebook);
    static const SoaCodebook kSoa4 = build_soa(kVq4dCodebook);
    return dim == 3 ? kSoa3 : kSoa4;
}

inline void fwht_inplace(float* x, size_t n) {
    for (size_t len = 1; len < n; len <<= 1) {
        const size_t step = len << 1;
        for (size_t i = 0; i < n; i += step) {
#if defined(__ARM_NEON) || defined(__aarch64__)
            if (len >= 4) {
                for (size_t j = 0; j < len; j += 4) {
                    const float32x4_t a = vld1q_f32(x + i + j);
                    const float32x4_t b = vld1q_f32(x + i + j + len);
                    vst1q_f32(x + i + j, vaddq_f32(a, b));
                    vst1q_f32(x + i + j + len, vsubq_f32(a, b));
                }
                continue;
            }
#endif
            for (size_t j = 0; j < len; ++j) {
                const float a = x[i + j];
                const float b = x[i + j + len];
                x[i + j] = a + b;
                x[i + j + len] = a - b;
            }
        }
    }
    const float inv_sqrt = (n == 64) ? 0.125F : (1.0F / std::sqrt(static_cast<float>(n)));
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float32x4_t sv = vdupq_n_f32(inv_sqrt);
    for (size_t i = 0; i < n; i += 4) {
        vst1q_f32(x + i, vmulq_f32(vld1q_f32(x + i), sv));
    }
#else
    for (size_t i = 0; i < n; ++i) x[i] *= inv_sqrt;
#endif
}

inline uint8_t nearest_antipodal(int32_t dim, const SoaCodebook& soa, float v0, float v1, float v2, float v3) {
    float best_score = 1e30F;
    int best_idx = 0;
    float best_dot = 0.0F;
#if defined(__ARM_NEON) || defined(__aarch64__)
    const float32x4_t q0 = vdupq_n_f32(v0);
    const float32x4_t q1 = vdupq_n_f32(v1);
    const float32x4_t q2 = vdupq_n_f32(v2);
    const float32x4_t q3 = vdupq_n_f32(v3);
    float32x4_t vmin = vdupq_n_f32(1e30F);
    uint32x4_t vidx = vdupq_n_u32(0);
    float32x4_t vdot_best = vdupq_n_f32(0.0F);
    const uint32x4_t base_lanes = {0u, 1u, 2u, 3u};
    for (uint32_t j = 0; j < 128; j += 4) {
        float32x4_t dot = vmulq_f32(q0, vld1q_f32(soa.c0 + j));
        dot = vfmaq_f32(dot, q1, vld1q_f32(soa.c1 + j));
        dot = vfmaq_f32(dot, q2, vld1q_f32(soa.c2 + j));
        if (dim == 4) dot = vfmaq_f32(dot, q3, vld1q_f32(soa.c3 + j));
        const float32x4_t score = vsubq_f32(vld1q_f32(soa.half_norm2 + j), vabsq_f32(dot));
        const uint32x4_t mask = vcltq_f32(score, vmin);
        vmin = vbslq_f32(mask, score, vmin);
        vidx = vbslq_u32(mask, vaddq_u32(vdupq_n_u32(j), base_lanes), vidx);
        vdot_best = vbslq_f32(mask, dot, vdot_best);
    }
    float scores[4], dots[4];
    uint32_t idxs[4];
    vst1q_f32(scores, vmin);
    vst1q_f32(dots, vdot_best);
    vst1q_u32(idxs, vidx);
    for (int lane = 0; lane < 4; ++lane) {
        if (scores[lane] < best_score) {
            best_score = scores[lane];
            best_idx = static_cast<int>(idxs[lane]);
            best_dot = dots[lane];
        }
    }
#else
    for (int j = 0; j < 128; ++j) {
        float dot = v0 * soa.c0[j] + v1 * soa.c1[j] + v2 * soa.c2[j];
        if (dim == 4) dot += v3 * soa.c3[j];
        const float score = soa.half_norm2[j] - std::fabs(dot);
        if (score < best_score) {
            best_score = score;
            best_idx = j;
            best_dot = dot;
        }
    }
#endif
    return static_cast<uint8_t>(best_idx | (best_dot < 0.0F ? 0x80 : 0x00));
}

} // namespace

const float (*vq3d_codebook())[4] { return kVq3dCodebook; }
const float (*vq4d_codebook())[4] { return kVq4dCodebook; }

void hadamard_transform_rows_f32(float* data, size_t rows, size_t cols, size_t block) {
    if (block == 0 || cols % block != 0) return;
    for (size_t r = 0; r < rows; ++r) {
        float* row = data + r * cols;
        for (size_t c = 0; c < cols; c += block) fwht_inplace(row + c, block);
    }
}

void hadamard_transform_cols_f32(float* data, size_t rows, size_t cols, size_t block) {
    if (block == 0 || rows % block != 0) return;
    const float inv_sqrt = (block == 64) ? 0.125F : (1.0F / std::sqrt(static_cast<float>(block)));
    for (size_t r0 = 0; r0 < rows; r0 += block) {
        for (size_t len = 1; len < block; len <<= 1) {
            const size_t step = len << 1;
            for (size_t i = 0; i < block; i += step) {
                for (size_t j = 0; j < len; ++j) {
                    float* ra = data + (r0 + i + j) * cols;
                    float* rb = data + (r0 + i + j + len) * cols;
                    size_t c = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
                    for (; c + 4 <= cols; c += 4) {
                        const float32x4_t a = vld1q_f32(ra + c);
                        const float32x4_t b = vld1q_f32(rb + c);
                        vst1q_f32(ra + c, vaddq_f32(a, b));
                        vst1q_f32(rb + c, vsubq_f32(a, b));
                    }
#endif
                    for (; c < cols; ++c) {
                        const float a = ra[c];
                        const float b = rb[c];
                        ra[c] = a + b;
                        rb[c] = a - b;
                    }
                }
            }
        }
        for (size_t i = 0; i < block; ++i) {
            float* row = data + (r0 + i) * cols;
            size_t c = 0;
#if defined(__ARM_NEON) || defined(__aarch64__)
            const float32x4_t sv = vdupq_n_f32(inv_sqrt);
            for (; c + 4 <= cols; c += 4) vst1q_f32(row + c, vmulq_f32(vld1q_f32(row + c), sv));
#endif
            for (; c < cols; ++c) row[c] *= inv_sqrt;
        }
    }
}

void vq_pack_row_block_f32(int32_t dim, size_t total_rows, size_t cols, size_t row_begin, size_t row_end,
                           const float* weights_row_slice, uint8_t* out_values, uint8_t* out_scales,
                           float* out_global) {
    const size_t n_rows = row_end - row_begin;
    const size_t group = vq_group_size(dim);
    const size_t k_pad = vq_padded_cols(dim, cols);
    const size_t n_groups = k_pad / group;
    const size_t row_codes = k_pad / static_cast<size_t>(dim);
    const size_t had_col = vq_col_hadamard_block(cols);
    const size_t had_row = vq_row_hadamard_block(total_rows);
    const float (*lut)[4] = dim == 3 ? kVq3dCodebook : kVq4dCodebook;
    const SoaCodebook& soa = soa_codebook(dim);

    // 1. Copy slice [n_rows, cols] into scratch, apply 2D Hadamard (cols then rows).
    std::vector<float> buf(n_rows * cols);
    std::memcpy(buf.data(), weights_row_slice, n_rows * cols * sizeof(float));
    if (had_col) hadamard_transform_rows_f32(buf.data(), n_rows, cols, had_col);
    if (had_row) hadamard_transform_cols_f32(buf.data(), n_rows, cols, had_row);

    // 2. Quantize each row in groups of 48 (VQ3D) or 64 (VQ4D) with exact VP-Scale and two-level UINT8/FP32 scales.
    std::vector<float> padded_row(k_pad, 0.0F);
    std::vector<float> row_scales(n_groups);
    for (size_t lr = 0; lr < n_rows; ++lr) {
        const size_t r = row_begin + lr;
        std::memcpy(padded_row.data(), buf.data() + lr * cols, cols * sizeof(float));
        if (k_pad > cols) std::fill(padded_row.begin() + static_cast<ptrdiff_t>(cols), padded_row.end(), 0.0F);

        uint8_t* dst_codes = out_values + r * row_codes;
        uint8_t* dst_scales = out_scales + r * n_groups;
        float max_scale = 0.0F;

        for (size_t g = 0; g < n_groups; ++g) {
            const float* blk = padded_row.data() + g * group;
            double ww = 0.0;
            for (size_t i = 0; i < group; ++i) ww += static_cast<double>(blk[i]) * static_cast<double>(blk[i]);
            const float rms = static_cast<float>(std::sqrt(std::max(ww / static_cast<double>(group), 1e-12)));
            const float inv_rms = 1.0F / rms;

            uint8_t* g_codes = dst_codes + g * 16;
            double qq = 0.0;
            if (dim == 3) {
                for (size_t v = 0; v < 16; ++v) {
                    const float x0 = blk[3 * v] * inv_rms;
                    const float x1 = blk[3 * v + 1] * inv_rms;
                    const float x2 = blk[3 * v + 2] * inv_rms;
                    const uint8_t code = nearest_antipodal(3, soa, x0, x1, x2, 0.0F);
                    g_codes[v] = code;
                    const float* c = lut[code & 0x7F];
                    qq += static_cast<double>(c[0]) * c[0] + static_cast<double>(c[1]) * c[1] +
                          static_cast<double>(c[2]) * c[2];
                }
            } else {
                for (size_t v = 0; v < 16; ++v) {
                    const float x0 = blk[4 * v] * inv_rms;
                    const float x1 = blk[4 * v + 1] * inv_rms;
                    const float x2 = blk[4 * v + 2] * inv_rms;
                    const float x3 = blk[4 * v + 3] * inv_rms;
                    const uint8_t code = nearest_antipodal(4, soa, x0, x1, x2, x3);
                    g_codes[v] = code;
                    const float* c = lut[code & 0x7F];
                    qq += static_cast<double>(c[0]) * c[0] + static_cast<double>(c[1]) * c[1] +
                          static_cast<double>(c[2]) * c[2] + static_cast<double>(c[3]) * c[3];
                }
            }
            // Exact Variance-Preserving Scale: s_vp = ||w||_2 / ||q||_2 (0 for zero blocks).
            const float s_vp = (ww > 1e-24 && qq > 1e-12) ? static_cast<float>(std::sqrt(ww / qq)) : 0.0F;
            row_scales[g] = s_vp;
            if (s_vp > max_scale) max_scale = s_vp;
        }

        const float g_step = max_scale > 1e-12F ? (max_scale / 255.0F) : 0.0F;
        out_global[r] = g_step;
        if (g_step > 0.0F) {
            const float inv_step = 1.0F / g_step;
            for (size_t g = 0; g < n_groups; ++g) {
                const int q = static_cast<int>(std::lround(row_scales[g] * inv_step));
                dst_scales[g] = static_cast<uint8_t>(std::clamp(q, 0, 255));
            }
        } else {
            std::memset(dst_scales, 0, n_groups);
        }
    }
}

void vq_linear_rotated_f32(int32_t dim, size_t tokens, size_t rows, size_t cols, const float* x_rot,
                           const uint8_t* values, const uint8_t* scales, const float* global, float* y_rot,
                           size_t row_begin, size_t row_end) {
    const size_t group = vq_group_size(dim);
    const size_t k_pad = vq_padded_cols(dim, cols);
    const size_t n_groups = k_pad / group;
    const size_t row_codes = k_pad / static_cast<size_t>(dim);
    const float (*lut)[4] = dim == 3 ? kVq3dCodebook : kVq4dCodebook;

    for (size_t r = row_begin; r < row_end; ++r) {
        const uint8_t* r_codes = values + r * row_codes;
        const uint8_t* r_scales = scales + r * n_groups;
        const float r_global = global[r];
        for (size_t t = 0; t < tokens; ++t) {
            const float* xr = x_rot + t * k_pad;
            float total = 0.0F;
            if (dim == 3) {
                for (size_t g = 0; g < n_groups; ++g) {
                    const uint8_t* gc = r_codes + g * 16;
                    const float* xb = xr + g * 48;
                    float gsum = 0.0F;
                    for (size_t v = 0; v < 16; ++v) {
                        const uint8_t code = gc[v];
                        const float* c = lut[code & 0x7F];
                        const float d = xb[3 * v] * c[0] + xb[3 * v + 1] * c[1] + xb[3 * v + 2] * c[2];
                        gsum += (code & 0x80) ? -d : d;
                    }
                    total += gsum * static_cast<float>(r_scales[g]);
                }
            } else {
                for (size_t g = 0; g < n_groups; ++g) {
                    const uint8_t* gc = r_codes + g * 16;
                    const float* xb = xr + g * 64;
#if defined(__ARM_NEON) || defined(__aarch64__)
                    float32x4_t acc = vdupq_n_f32(0.0F);
                    for (size_t v = 0; v < 16; ++v) {
                        const uint8_t code = gc[v];
                        float32x4_t w = vld1q_f32(lut[code & 0x7F]);
                        if (code & 0x80) w = vnegq_f32(w);
                        acc = vfmaq_f32(acc, vld1q_f32(xb + 4 * v), w);
                    }
                    total += vaddvq_f32(acc) * static_cast<float>(r_scales[g]);
#else
                    float gsum = 0.0F;
                    for (size_t v = 0; v < 16; ++v) {
                        const uint8_t code = gc[v];
                        const float* c = lut[code & 0x7F];
                        const float d =
                            xb[4 * v] * c[0] + xb[4 * v + 1] * c[1] + xb[4 * v + 2] * c[2] + xb[4 * v + 3] * c[3];
                        gsum += (code & 0x80) ? -d : d;
                    }
                    total += gsum * static_cast<float>(r_scales[g]);
#endif
                }
            }
            y_rot[t * rows + r] = total * r_global;
        }
    }
}

} // namespace ynn
