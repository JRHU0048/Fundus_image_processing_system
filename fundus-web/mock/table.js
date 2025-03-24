// 这段代码利用 Mock.js 生成了模拟的表格数据，并模拟了一个 API 接口，
// 当客户端发起 GET 请求到 /vue-admin-template/table/list 时，会返回包含这些模拟数据的响应。

// 引入 Mock.js 库，该库可用于生成随机数据和模拟 API 响应
const Mock = require('mockjs');

// 使用 Mock.js 的 mock 方法生成模拟数据
const data = Mock.mock({
    // 生成一个包含 30 个元素的数组，数组元素的结构由后面的对象定义
    'items|30': [{
        // 生成一个唯一的 ID
        id: '@id',
        // 生成一个包含 10 到 20 个单词的随机句子作为标题
        title: '@sentence(10, 20)',
        // 随机从 'published'、'draft'、'deleted' 中选取一个作为状态
        'status|1': ['published', 'draft', 'deleted'],
        // 固定值 'name' 作为作者
        author: 'name',
        // 生成一个随机的日期和时间字符串作为显示时间
        display_time: '@datetime',
        // 生成一个 300 到 5000 之间的随机整数作为页面浏览量
        pageviews: '@integer(300, 5000)'
    }]
});

// 导出一个数组，数组中的每个元素代表一个模拟的 API 接口
module.exports = [
    {
        // 模拟接口的 URL
        url: '/vue-admin-template/table/list',
        // 模拟接口的请求方法
        type: 'get',
        // 模拟接口的响应处理函数，接收 config 参数（通常包含请求的配置信息）
        response: config => {
            // 从之前生成的模拟数据中获取 items 数组
            const items = data.items;
            // 返回一个包含状态码和数据的对象作为响应
            return {
                // 状态码，20000 通常表示请求成功
                code: 20000,
                data: {
                    // 数据总数，即 items 数组的长度
                    total: items.length,
                    // 具体的数据项，即 items 数组
                    items: items
                }
            };
        }
    }
];