import Hangul from 'hangul-js';

const wansungChars =
    '가각간갇갈갉갊감갑값갓갔강갖갗같갚갛개객갠갤갬갭갯갰갱갸갹갼걀걋걍걔걘걜거걱건걷걸걺검겁것겄겅겆겉겊겋게겐겔겜겝겟겠겡겨격겪견겯결겸겹겻겼경곁계곈곌곕곗고곡곤곧골곪곬곯곰곱곳공곶과곽관괄괆괌괍괏광괘괜괠괩괬괭괴괵괸괼굄굅굇굉교굔굘굡굣구국군굳굴굵굶굻굼굽굿궁궂궈궉권궐궜궝궤궷귀귁귄귈귐귑귓규균귤그극근귿글긁금급긋긍긔기긱긴긷길긺김깁깃깅깆깊까깍깎깐깔깖깜깝깟깠깡깥깨깩깬깰깸깹깻깼깽꺄꺅꺌꺼꺽꺾껀껄껌껍껏껐껑께껙껜껨껫껭껴껸껼꼇꼈꼍꼐꼬꼭꼰꼲꼴꼼꼽꼿꽁꽂꽃꽈꽉꽐꽜꽝꽤꽥꽹꾀꾄꾈꾐꾑꾕꾜꾸꾹꾼꿀꿇꿈꿉꿋꿍꿎꿔꿜꿨꿩꿰꿱꿴꿸뀀뀁뀄뀌뀐뀔뀜뀝뀨끄끅끈끊끌끎끓끔끕끗끙끝끼끽낀낄낌낍낏낑나낙낚난낟날낡낢남납낫났낭낮낯낱낳내낵낸낼냄냅냇냈냉냐냑냔냘냠냥너넉넋넌널넒넓넘넙넛넜넝넣네넥넨넬넴넵넷넸넹녀녁년녈념녑녔녕녘녜녠노녹논놀놂놈놉놋농높놓놔놘놜놨뇌뇐뇔뇜뇝뇟뇨뇩뇬뇰뇹뇻뇽누눅눈눋눌눔눕눗눙눠눴눼뉘뉜뉠뉨뉩뉴뉵뉼늄늅늉느늑는늘늙늚늠늡늣능늦늪늬늰늴니닉닌닐닒님닙닛닝닢다닥닦단닫달닭닮닯닳담답닷닸당닺닻닿대댁댄댈댐댑댓댔댕댜더덕덖던덛덜덞덟덤덥덧덩덫덮데덱덴델뎀뎁뎃뎄뎅뎌뎐뎔뎠뎡뎨뎬도독돈돋돌돎돐돔돕돗동돛돝돠돤돨돼됐되된될됨됩됫됴두둑둔둘둠둡둣둥둬뒀뒈뒝뒤뒨뒬뒵뒷뒹듀듄듈듐듕드득든듣들듦듬듭듯등듸디딕딘딛딜딤딥딧딨딩딪따딱딴딸땀땁땃땄땅땋때땍땐땔땜땝땟땠땡떠떡떤떨떪떫떰떱떳떴떵떻떼떽뗀뗄뗌뗍뗏뗐뗑뗘뗬또똑똔똘똥똬똴뙈뙤뙨뚜뚝뚠뚤뚫뚬뚱뛔뛰뛴뛸뜀뜁뜅뜨뜩뜬뜯뜰뜸뜹뜻띄띈띌띔띕띠띤띨띰띱띳띵라락란랄람랍랏랐랑랒랖랗래랙랜랠램랩랫랬랭랴략랸럇량러럭런럴럼럽럿렀렁렇레렉렌렐렘렙렛렝려력련렬렴렵렷렸령례롄롑롓로록론롤롬롭롯롱롸롼뢍뢨뢰뢴뢸룀룁룃룅료룐룔룝룟룡루룩룬룰룸룹룻룽뤄뤘뤠뤼뤽륀륄륌륏륑류륙륜률륨륩륫륭르륵른를름릅릇릉릊릍릎리릭린릴림립릿링마막만많맏말맑맒맘맙맛망맞맡맣매맥맨맬맴맵맷맸맹맺먀먁먈먕머먹먼멀멂멈멉멋멍멎멓메멕멘멜멤멥멧멨멩며멱면멸몃몄명몇몌모목몫몬몰몲몸몹못몽뫄뫈뫘뫙뫼묀묄묍묏묑묘묜묠묩묫무묵묶문묻물묽묾뭄뭅뭇뭉뭍뭏뭐뭔뭘뭡뭣뭬뮈뮌뮐뮤뮨뮬뮴뮷므믄믈믐믓미믹민믿밀밂밈밉밋밌밍및밑바박밖밗반받발밝밞밟밤밥밧방밭배백밴밸뱀뱁뱃뱄뱅뱉뱌뱍뱐뱝버벅번벋벌벎범법벗벙벚베벡벤벧벨벰벱벳벴벵벼벽변별볍볏볐병볕볘볜보복볶본볼봄봅봇봉봐봔봤봬뵀뵈뵉뵌뵐뵘뵙뵤뵨부북분붇불붉붊붐붑붓붕붙붚붜붤붰붸뷔뷕뷘뷜뷩뷰뷴뷸븀븃븅브븍븐블븜븝븟비빅빈빌빎빔빕빗빙빚빛빠빡빤빨빪빰빱빳빴빵빻빼빽뺀뺄뺌뺍뺏뺐뺑뺘뺙뺨뻐뻑뻔뻗뻘뻠뻣뻤뻥뻬뼁뼈뼉뼘뼙뼛뼜뼝뽀뽁뽄뽈뽐뽑뽕뾔뾰뿅뿌뿍뿐뿔뿜뿟뿡쀼쁑쁘쁜쁠쁨쁩삐삑삔삘삠삡삣삥사삭삯산삳살삵삶삼삽삿샀상샅새색샌샐샘샙샛샜생샤샥샨샬샴샵샷샹섀섄섈섐섕서석섞섟선섣설섦섧섬섭섯섰성섶세섹센셀셈셉셋셌셍셔셕션셜셤셥셧셨셩셰셴셸솅소속솎손솔솖솜솝솟송솥솨솩솬솰솽쇄쇈쇌쇔쇗쇘쇠쇤쇨쇰쇱쇳쇼쇽숀숄숌숍숏숑수숙순숟술숨숩숫숭숯숱숲숴쉈쉐쉑쉔쉘쉠쉥쉬쉭쉰쉴쉼쉽쉿슁슈슉슐슘슛슝스슥슨슬슭슴습슷승시식신싣실싫심십싯싱싶싸싹싻싼쌀쌈쌉쌌쌍쌓쌔쌕쌘쌜쌤쌥쌨쌩썅써썩썬썰썲썸썹썼썽쎄쎈쎌쏀쏘쏙쏜쏟쏠쏢쏨쏩쏭쏴쏵쏸쐈쐐쐤쐬쐰쐴쐼쐽쑈쑤쑥쑨쑬쑴쑵쑹쒀쒔쒜쒸쒼쓩쓰쓱쓴쓸쓺쓿씀씁씌씐씔씜씨씩씬씰씸씹씻씽아악안앉않알앍앎앓암압앗았앙앝앞애액앤앨앰앱앳앴앵야약얀얄얇얌얍얏양얕얗얘얜얠얩어억언얹얻얼얽얾엄업없엇었엉엊엌엎에엑엔엘엠엡엣엥여역엮연열엶엷염엽엾엿였영옅옆옇예옌옐옘옙옛옜오옥온올옭옮옰옳옴옵옷옹옻와왁완왈왐왑왓왔왕왜왝왠왬왯왱외왹왼욀욈욉욋욍요욕욘욜욤욥욧용우욱운울욹욺움웁웃웅워웍원월웜웝웠웡웨웩웬웰웸웹웽위윅윈윌윔윕윗윙유육윤율윰윱윳융윷으윽은을읊음읍읏응읒읓읔읕읖읗의읜읠읨읫이익인일읽읾잃임입잇있잉잊잎자작잔잖잗잘잚잠잡잣잤장잦재잭잰잴잼잽잿쟀쟁쟈쟉쟌쟎쟐쟘쟝쟤쟨쟬저적전절젊점접젓정젖제젝젠젤젬젭젯젱져젼졀졈졉졌졍졔조족존졸졺좀좁좃종좆좇좋좌좍좔좝좟좡좨좼좽죄죈죌죔죕죗죙죠죡죤죵주죽준줄줅줆줌줍줏중줘줬줴쥐쥑쥔쥘쥠쥡쥣쥬쥰쥴쥼즈즉즌즐즘즙즛증지직진짇질짊짐집짓징짖짙짚짜짝짠짢짤짧짬짭짯짰짱째짹짼쨀쨈쨉쨋쨌쨍쨔쨘쨩쩌쩍쩐쩔쩜쩝쩟쩠쩡쩨쩽쪄쪘쪼쪽쫀쫄쫌쫍쫏쫑쫓쫘쫙쫠쫬쫴쬈쬐쬔쬘쬠쬡쭁쭈쭉쭌쭐쭘쭙쭝쭤쭸쭹쮜쮸쯔쯤쯧쯩찌찍찐찔찜찝찡찢찧차착찬찮찰참찹찻찼창찾채책챈챌챔챕챗챘챙챠챤챦챨챰챵처척천철첨첩첫첬청체첵첸첼쳄쳅쳇쳉쳐쳔쳤쳬쳰촁초촉촌촐촘촙촛총촤촨촬촹최쵠쵤쵬쵭쵯쵱쵸춈추축춘출춤춥춧충춰췄췌췐취췬췰췸췹췻췽츄츈츌츔츙츠측츤츨츰츱츳층치칙친칟칠칡침칩칫칭카칵칸칼캄캅캇캉캐캑캔캘캠캡캣캤캥캬캭컁커컥컨컫컬컴컵컷컸컹케켁켄켈켐켑켓켕켜켠켤켬켭켯켰켱켸코콕콘콜콤콥콧콩콰콱콴콸쾀쾅쾌쾡쾨쾰쿄쿠쿡쿤쿨쿰쿱쿳쿵쿼퀀퀄퀑퀘퀭퀴퀵퀸퀼큄큅큇큉큐큔큘큠크큭큰클큼큽킁키킥킨킬킴킵킷킹타탁탄탈탉탐탑탓탔탕태택탠탤탬탭탯탰탱탸턍터턱턴털턺텀텁텃텄텅테텍텐텔템텝텟텡텨텬텼톄톈토톡톤톨톰톱톳통톺톼퇀퇘퇴퇸툇툉툐투툭툰툴툼툽툿퉁퉈퉜퉤튀튁튄튈튐튑튕튜튠튤튬튱트특튼튿틀틂틈틉틋틔틘틜틤틥티틱틴틸팀팁팃팅파팍팎판팔팖팜팝팟팠팡팥패팩팬팰팸팹팻팼팽퍄퍅퍼퍽펀펄펌펍펏펐펑페펙펜펠펨펩펫펭펴편펼폄폅폈평폐폘폡폣포폭폰폴폼폽폿퐁퐈퐝푀푄표푠푤푭푯푸푹푼푿풀풂품풉풋풍풔풩퓌퓐퓔퓜퓟퓨퓬퓰퓸퓻퓽프픈플픔픕픗피픽핀필핌핍핏핑하학한할핥함합핫항해핵핸핼햄햅햇했행햐향허헉헌헐헒험헙헛헝헤헥헨헬헴헵헷헹혀혁현혈혐협혓혔형혜혠혤혭호혹혼홀홅홈홉홋홍홑화확환활홧황홰홱홴횃횅회획횐횔횝횟횡효횬횰횹횻후훅훈훌훑훔훗훙훠훤훨훰훵훼훽휀휄휑휘휙휜휠휨휩휫휭휴휵휸휼흄흇흉흐흑흔흖흗흘흙흠흡흣흥흩희흰흴흼흽힁히힉힌힐힘힙힛힝';

const wansungCharsDic = {};
for (let c of wansungChars.split('')) {
    wansungCharsDic[c] = 1;
}

const isWansungChar = (c) => wansungCharsDic.hasOwnProperty(c);

const componentConversionTable = [
    {
        ㄱ: 'ㄱㄲㅋ',
        ㄲ: 'ㄲㅋ',
        ㄴ: 'ㄴ',
        ㄷ: 'ㄷㄸㅌ',
        ㄸ: 'ㄸㅌ',
        ㄹ: 'ㄹ',
        ㅁ: 'ㅁ',
        ㅂ: 'ㅂㅃㅍ',
        ㅃ: 'ㅃㅍ',
        ㅅ: 'ㅅㅆ',
        ㅆ: 'ㅆㅅ',
        ㅇ: 'ㅇ',
        ㅈ: 'ㅈㅉㅊ',
        ㅉ: 'ㅉㅊ',
        ㅊ: 'ㅊㅉ',
        ㅋ: 'ㅋㄲ',
        ㅌ: 'ㅌㄸ',
        ㅍ: 'ㅍㅃ',
        ㅎ: 'ㅎ',
    },
    {
        ㅏ: 'ㅏㅑ',
        ㅐ: 'ㅐㅒㅔ',
        ㅑ: 'ㅑㅏ',
        ㅒ: 'ㅒㅖ',
        ㅓ: 'ㅓㅕ',
        ㅔ: 'ㅔㅖㅞ',
        ㅕ: 'ㅕㅓ',
        ㅖ: 'ㅖㅒ',
        ㅗ: 'ㅗㅛ',
        ㅘ: 'ㅘ',
        ㅙ: 'ㅙㅚㅞ',
        ㅚ: 'ㅚㅙㅞ',
        ㅛ: 'ㅛㅗ',
        ㅜ: 'ㅜㅠ',
        ㅝ: 'ㅝㅓ',
        ㅞ: 'ㅞㅚㅙㅔ',
        ㅟ: 'ㅟ',
        ㅠ: 'ㅠㅜ',
        ㅡ: 'ㅡㅜ',
        ㅢ: 'ㅢㅟ',
        ㅣ: 'ㅣㅟ',
    },
    {
        x: 'x',
        ㄱ: 'ㄱㄳㅋ',
        ㄲ: 'ㄲㅋ',
        ㄳ: 'ㄳㄱㅋ',
        ㄴ: 'ㄴㄵㄶ',
        ㄵ: 'ㄵㄶㄴ',
        ㄶ: 'ㄶㄵㄴ',
        ㄷ: 'ㄷㅌㅅㅆㅈㅊ',
        ㄹ: 'ㄹㄺㄻㄼㄽㄾㄿㅀ',
        ㄺ: 'ㄺㄹㄻㄼㄽㄾㄿㅀ',
        ㄻ: 'ㄻㄹㄺㄼㄽㄾㄿㅀ',
        ㄼ: 'ㄼㄹㄺㄻㄽㄾㄿㅀ',
        ㄽ: 'ㄽㄹㄺㄻㄼㄾㄿㅀ',
        ㄾ: 'ㄾㄹㄺㄻㄼㄽㄿㅀ',
        ㄿ: 'ㄿㄹㄺㄻㄼㄽㄾㅀ',
        ㅀ: 'ㅀㄹㄺㄻㄼㄽㄾㄿ',
        ㅁ: 'ㅁㄻ',
        ㅂ: 'ㅂㅄㅍ',
        ㅄ: 'ㅄㅂㅍ',
        ㅅ: 'ㅅㄷㅌㅆㅈㅊ',
        ㅆ: 'ㅆㄷㅌㅅㅈㅊ',
        ㅇ: 'ㅇㅎ',
        ㅈ: 'ㅈㄷㅌㅅㅆㅊ',
        ㅊ: 'ㅊㄷㅌㅅㅆㅈ',
        ㅋ: 'ㅋㄱㄳ',
        ㅌ: 'ㅌㄷㅅㅆㅈㅊ',
        ㅍ: 'ㅍㅂㅄ',
        ㅎ: 'ㅎ',
    },
];

const extraEndSoundCandidates = [
    'ㄱ',
    'ㄲ',
    'ㄳ',
    'ㄴ',
    'ㄵ',
    'ㄶ',
    'ㄷ',
    'ㄹ',
    'ㄺ',
    'ㄻ',
    'ㄼ',
    'ㄽ',
    'ㄾ',
    'ㄿ',
    'ㅀ',
    'ㅁ',
    'ㅂ',
    'ㅄ',
    'ㅅ',
    'ㅆ',
    'ㅇ',
    'ㅈ',
    'ㅊ',
    'ㅋ',
    'ㅌ',
    'ㅍ',
    'ㅎ',
];

const divideIntoComponents = (c) => {
    const arr = Hangul.d(c);
    const midIndex = arr.findIndex((elem) => Hangul.isVowel(elem));
    const index = arr
        .slice(midIndex)
        .findIndex((elem) => Hangul.isConsonant(elem));
    const lastIndex = index > 0 ? midIndex + index : arr.length;

    const first = Hangul.a(arr.slice(0, midIndex));
    const mid = Hangul.a(arr.slice(midIndex, lastIndex));
    let last = Hangul.a(arr.slice(lastIndex));
    if (last === '') {
        last = 'x'; // Expresses no end component.
    }
    return [first, mid, last];
};

const arrayShuffle = (a) => {
    let j, x, i;
    for (i = a.length; i; i--) {
        j = Math.floor(Math.random() * i);
        x = a[i - 1];
        a[i - 1] = a[j];
        a[j] = x;
    }
    return a;
};

const uglifyHangul = (
    c,
    componentChangeDegree,
    addExtraEndSoundDegree,
    onlyWansungChars
) => {
    const components = divideIntoComponents(c);

    const numOfMaxPossibleComponentChanges = Math.ceil(
        (componentChangeDegree * 3) / 100
    );
    const componentIndicesAllowedToChange = arrayShuffle([0, 1, 2]).slice(
        0,
        numOfMaxPossibleComponentChanges
    );

    for (let i = 0; i < 10; i++) {
        // Attempt up to 10 times to find a wansung char.
        const arr = [];
        for (let j = 0; j < components.length; j++) {
            const org = components[j];
            const ct = componentConversionTable[j][org];
            let converted;
            if (componentIndicesAllowedToChange.includes(j)) {
                converted = ct[Math.floor(Math.random() * ct.length)];
            } else {
                converted = org;
            }
            //            console.log(i, j, org, ct, converted);
            if (converted === 'x') {
                if (Math.random() * 100 < addExtraEndSoundDegree) {
                    converted =
                        extraEndSoundCandidates[
                            Math.floor(
                                Math.random() * extraEndSoundCandidates.length
                            )
                        ];
                    arr.push(converted);
                }
            } else {
                arr.push(converted);
            }
        }
        const convertedChar = Hangul.a(arr);
        if (c !== convertedChar) {
            if (onlyWansungChars) {
                if (isWansungChar(convertedChar)) {
                    return convertedChar;
                }
            } else {
                return convertedChar;
            }
        }
    }

    return c;
};

const convertToAsItSounds = (charArr, asItSoundsDegree, onlyWansungChars) => {
    const endPlusHConversionTable = {
        ㄱ: 'ㅋ',
        //        "ㄷ": "ㅌ",
        ㅂ: 'ㅍ',
        //        "ㅅ": "ㅌ",
        //        "ㅈ": "ㅊ",
        //        "ㅊ": "ㅌ",
    };
    for (let i = 1; i < charArr.length; i++) {
        const c1 = charArr[i - 1];
        const c2 = charArr[i];
        if (Hangul.isHangul(c1) && Hangul.isHangul(c2)) {
            if (Math.random() * 100 >= asItSoundsDegree) {
                continue;
            }
            const c1cs = divideIntoComponents(c1);
            const c2cs = divideIntoComponents(c2);
            const c1end = c1cs[2];
            const c2beg = c2cs[0];
            if (c1end !== 'x' && c1end !== 'ㅇ' && c2beg === 'ㅇ') {
                const c1endArr = Hangul.d(c1end);
                c2cs[0] = c1endArr.pop();
                if (c2cs[2] === 'x') {
                    c2cs.pop();
                }
                const newC1 = Hangul.a([c1cs[0], c1cs[1], ...c1endArr]);
                const newC2 = Hangul.a(c2cs);
                if (onlyWansungChars) {
                    if (isWansungChar(newC1) && isWansungChar(newC2)) {
                        charArr[i - 1] = newC1;
                        charArr[i] = newC2;
                    }
                } else {
                    charArr[i - 1] = newC1;
                    charArr[i] = newC2;
                }
            }
            if (
                endPlusHConversionTable.hasOwnProperty(c1end) &&
                c2beg === 'ㅎ'
            ) {
                c1cs.pop();
                c2cs[0] = endPlusHConversionTable[c1end];
                if (c2cs[2] === 'x') {
                    c2cs.pop();
                }
                const newC1 = Hangul.a(c1cs);
                const newC2 = Hangul.a(c2cs);
                if (onlyWansungChars) {
                    if (isWansungChar(newC1) && isWansungChar(newC2)) {
                        charArr[i - 1] = newC1;
                        charArr[i] = newC2;
                    }
                } else {
                    charArr[i - 1] = newC1;
                    charArr[i] = newC2;
                }
            }
        }
    }
};

const doubleConsonantAsEndSound = (
    charArr,
    doubleConsonantAsEndSoundDegree,
    onlyWansungChars
) => {
    const begToEndAsDoubleConsonantTable = {
        ㄱ: 'ㄱ',
        ㄲ: 'ㄱ',
        ㄴ: 'ㄴ',
        ㄷ: 'ㄷ',
        ㄸ: 'ㄷ',
        ㄹ: 'ㄹ',
        ㅁ: 'ㅁ',
        ㅂ: 'ㅂ',
        ㅃ: 'ㅂ',
        ㅅ: 'ㅅ',
        ㅆ: 'ㅅ',
        ㅇ: null,
        ㅈ: 'ㅈ',
        ㅉ: 'ㅈ',
        ㅊ: 'ㅊ',
        ㅋ: 'ㄱ',
        ㅌ: 'ㄷ',
        ㅍ: 'ㅂ',
        ㅎ: 'ㅎ',
    };
    for (let i = 1; i < charArr.length; i++) {
        const c1 = charArr[i - 1];
        const c2 = charArr[i];
        if (Hangul.isHangul(c1) && Hangul.isHangul(c2)) {
            if (Math.random() * 100 >= doubleConsonantAsEndSoundDegree) {
                continue;
            }
            const c1cs = divideIntoComponents(c1);
            const c2cs = divideIntoComponents(c2);
            const c1end = c1cs[2];
            const c2beg = c2cs[0];
            if (c1end === 'x') {
                const endAsDoubleConsonant =
                    begToEndAsDoubleConsonantTable[c2beg];
                if (endAsDoubleConsonant !== null) {
                    const newC1 = Hangul.a([
                        c1cs[0],
                        c1cs[1],
                        endAsDoubleConsonant,
                    ]);
                    if (onlyWansungChars) {
                        if (isWansungChar(newC1)) {
                            charArr[i - 1] = newC1;
                        }
                    } else {
                        charArr[i - 1] = newC1;
                    }
                }
            }
        }
    }
};

const uglifyChar = (
    c,
    componentChangeDegree,
    addExtraEndSoundDegree,
    onlyWansungChars
) => {
    if (Hangul.isHangul(c)) {
        return uglifyHangul(
            c,
            componentChangeDegree,
            addExtraEndSoundDegree,
            onlyWansungChars
        );
    } else {
        return c;
    }
};

export const uglifyText = (
    text,
    asItSoundsDegree,
    doubleConsonantAsEndSoundDegree,
    componentChangeDegree,
    addExtraEndSoundDegree,
    onlyWansungChars
) => {
    const charArr = [];
    for (let c of text.split('')) {
        charArr.push(c);
    }
    convertToAsItSounds(charArr, asItSoundsDegree, onlyWansungChars);
    doubleConsonantAsEndSound(
        charArr,
        doubleConsonantAsEndSoundDegree,
        onlyWansungChars
    );
    for (let i = 0; i < charArr.length; i++) {
        charArr[i] = uglifyChar(
            charArr[i],
            componentChangeDegree,
            addExtraEndSoundDegree,
            onlyWansungChars
        );
    }
    return charArr.join('');
};
