// 从 runjs 模块引入 run 函数，用于执行 shell 命令
const { run } = require('runjs');
// 引入 chalk 模块，用于美化控制台输出，使其带有颜色
const chalk = require('chalk');
// 引入 Vue 项目的配置文件 vue.config.js，以便获取项目配置信息
const config = require('../vue.config.js');
// 获取命令行参数，去除前两个参数（Node.js 可执行文件路径和当前执行的 JavaScript 文件路径）
const rawArgv = process.argv.slice(2);
// 将获取到的命令行参数用空格连接成一个字符串
const args = rawArgv.join(' ');

// 检查是否设置了预览选项，可通过环境变量 npm_config_preview 或者命令行参数 --preview 开启
if (process.env.npm_config_preview || rawArgv.includes('--preview')) {
    // 检查是否设置了 --report 参数，该参数通常用于生成构建报告
    const report = rawArgv.includes('--report');

    // 执行 vue-cli-service build 命令来构建 Vue 项目，并传递之前获取的命令行参数
    run(`vue-cli-service build ${args}`);

    // 设置静态服务器的端口号为 80
    const port = 80;
    // 从 vue.config.js 中获取项目的公共路径
    const publicPath = config.publicPath;

    // 引入 connect 模块，它是一个 Node.js 的中间件框架
    var connect = require('connect');
    // 引入 serve-static 模块，用于提供静态文件服务
    var serveStatic = require('serve-static');
    // 创建一个 connect 应用实例
    const app = connect();

    // 使用 serve-static 中间件，将 ./dist 目录下的静态文件映射到 publicPath 路径
    // 并指定默认的索引文件为 index.html 或者 /
    app.use(
        publicPath,
        serveStatic('./dist', {
            index: ['index.html', '/']
        })
    );

    // 启动静态服务器，监听指定的端口
    app.listen(port, function () {
        // 使用 chalk 模块输出绿色的控制台信息，显示预览地址
        console.log(chalk.green(`> Preview at  http://localhost:${port}${publicPath}`));
        // 如果设置了 --report 参数，输出构建报告的地址
        if (report) {
            console.log(chalk.green(`> Report at  http://localhost:${port}${publicPath}report.html`));
        }
    });
} else {
    // 若未设置预览选项，仅执行 vue-cli-service build 命令来构建 Vue 项目
    run(`vue-cli-service build ${args}`);
}