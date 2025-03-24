// 这段代码的主要功能是为 Express 应用搭建一个模拟服务，
// 通过监听 mock 目录下文件的变化，实现模拟接口的热更新。同时，使用 Mock.js 生成模拟数据，方便前端开发时进行接口调试。

// 引入 chokidar 模块，用于监听文件变化
const chokidar = require('chokidar');
// 引入 body-parser 模块，用于解析请求体
const bodyParser = require('body-parser');
// 引入 chalk 模块，用于美化控制台输出
const chalk = require('chalk');
// 引入 path 模块，用于处理文件路径
const path = require('path');
// 引入 Mock.js 模块，用于生成模拟数据
const Mock = require('mockjs');

// 获取 mock 目录的绝对路径，process.cwd() 返回当前 Node.js 进程的工作目录
const mockDir = path.join(process.cwd(), 'mock');

/**
 * 注册模拟路由到 Express 应用中
 * @param {Object} app - Express 应用实例
 * @returns {Object} - 包含模拟路由数量和起始索引的对象
 */
function registerRoutes(app) {
    let mockLastIndex;
    // 从 index.js 文件中导入模拟路由配置
    const { mocks } = require('./index.js');
    // 将模拟路由配置转换为适合服务器使用的格式
    const mocksForServer = mocks.map(route => {
        return responseFake(route.url, route.type, route.response);
    });
    // 遍历转换后的模拟路由配置，并将其注册到 Express 应用中
    for (const mock of mocksForServer) {
        app[mock.type](mock.url, mock.response);
        // 记录最后一个模拟路由在路由栈中的索引
        mockLastIndex = app._router.stack.length;
    }
    // 获取模拟路由的数量
    const mockRoutesLength = Object.keys(mocksForServer).length;
    return {
        mockRoutesLength: mockRoutesLength,
        mockStartIndex: mockLastIndex - mockRoutesLength
    };
}

/**
 * 注销模拟路由并清除缓存
 */
function unregisterRoutes() {
    // 遍历 require.cache 对象中的所有缓存模块
    Object.keys(require.cache).forEach(i => {
        // 如果缓存模块的路径包含 mock 目录
        if (i.includes(mockDir)) {
            // 从缓存中删除该模块
            delete require.cache[require.resolve(i)];
        }
    });
}

/**
 * 生成模拟响应对象
 * @param {string} url - 模拟接口的 URL
 * @param {string} type - 请求类型，如 'get'、'post' 等
 * @param {Function|Object} respond - 响应处理函数或响应数据
 * @returns {Object} - 包含 URL、请求类型和响应处理函数的对象
 */
// for mock server
const responseFake = (url, type, respond) => {
    return {
        // 使用正则表达式匹配 URL，拼接上环境变量中的基础 API 路径
        url: new RegExp(`${process.env.VUE_APP_BASE_API}${url}`),
        // 请求类型，默认为 'get'
        type: type || 'get',
        response(req, res) {
            // 打印请求的路径
            console.log('request invoke:' + req.path);
            // 使用 Mock.js 生成模拟数据并返回给客户端
            res.json(Mock.mock(respond instanceof Function ? respond(req, res) : respond));
        }
    };
};

/**
 * 主函数，用于初始化 Express 应用的模拟服务
 * @param {Object} app - Express 应用实例
 */
module.exports = app => {
    // 使用 body-parser 中间件解析 JSON 格式的请求体
    app.use(bodyParser.json());
    // 使用 body-parser 中间件解析 URL 编码的请求体
    app.use(bodyParser.urlencoded({
        extended: true
    }));

    // 注册模拟路由并获取相关信息
    const mockRoutes = registerRoutes(app);
    var mockRoutesLength = mockRoutes.mockRoutesLength;
    var mockStartIndex = mockRoutes.mockStartIndex;

    // 使用 chokidar 监听 mock 目录下的文件变化
    chokidar.watch(mockDir, {
        // 忽略 mock-server 目录下的文件
        ignored: /mock-server/,
        // 不监听初始的文件事件
        ignoreInitial: true
    }).on('all', (event, path) => {
        // 当文件发生变化或新增时
        if (event === 'change' || event === 'add') {
            try {
                // 从路由栈中移除旧的模拟路由
                app._router.stack.splice(mockStartIndex, mockRoutesLength);

                // 清除模拟路由的缓存
                unregisterRoutes();

                // 重新注册模拟路由并获取相关信息
                const mockRoutes = registerRoutes(app);
                mockRoutesLength = mockRoutes.mockRoutesLength;
                mockStartIndex = mockRoutes.mockStartIndex;

                // 打印模拟服务器热更新成功的信息
                console.log(chalk.magentaBright(`\n > Mock Server hot reload success! changed  ${path}`));
            } catch (error) {
                // 打印错误信息
                console.log(chalk.redBright(error));
            }
        }
    });
};